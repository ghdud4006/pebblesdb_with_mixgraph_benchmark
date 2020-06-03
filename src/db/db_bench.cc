//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#include <cassert>
#include <iostream>
#include <string>
#include <cstring>
#include <sstream>
#include <cstdlib>
#include <sys/mman.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <fstream>
#include <cmath>
#include <inttypes.h>

#include <sys/types.h>
#include <stdio.h>
#include <stdlib.h>
#include "db/db_impl.h"
#include "db/version_set.h"
#include "pebblesdb/cache.h"
#include "pebblesdb/db.h"
#include "pebblesdb/env.h"
#include "pebblesdb/write_batch.h"
#include "port/port.h"
#include "util/crc32c.h"
#include "util/histogram.h"
#include "util/mutexlock.h"
#include "util/random.h"
#include "util/testutil.h"
#include "util/testharness.h"
#include "util/gflags_compat.h"

#define MAX_TRACE_OPS 100000000
#define MAX_VALUE_SIZE (1024 * 1024)
#define sassert(X) {if (!(X)) std::cerr << "\n\n\n\n" << status.ToString() << "\n\n\n\n"; assert(X);}

#ifdef TIMER_LOG
	#define micros(a) a = Env::Default()->NowMicros()
	#define print_timer_info(a, b, c)   printf("%s: %lu micros (%f ms)\n", a, abs(b - c), abs(b - c)/1000.0);
#else
	#define micros(a)
	#define print_timer_info(a, b, c)
#endif


// Comma-separated list of operations to run in the specified order
//   Actual benchmarks:
//      fillseq       -- write N values in sequential key order in async mode
//      fillrandom    -- write N values in random key order in async mode
//      overwrite     -- overwrite N values in random key order in async mode
//      fillsync      -- write N/100 values in random key order in sync mode
//      fill100K      -- write N/1000 100K values in random order in async mode
//      deleteseq     -- delete N keys in sequential order
//      deleterandom  -- delete N keys in random order
//      readseq       -- read N times sequentially
//      readreverse   -- read N times in reverse order
//      readrandom    -- read N times in random order
//      readmissing   -- read N missing keys in random order
//      readhot       -- read N times in random order from 1% section of DB
//      seekrandom    -- N random seeks
//      crc32c        -- repeated crc32c of 4K of data
//      acquireload   -- load N*1000 times
//   Meta operations:
//      compact     -- Compact the entire DB
//      stats       -- Print DB stats
//      sstables    -- Print sstable info
//      heapprofile -- Dump a heap profile (if supported by this port)
static const char* FLAGS_benchmarks =
    "fillseq,"
    "fillsync,"
    "fillrandom,"
    "overwrite,"
    "readrandom,"
    "readrandom,"  // Extra run to allow previous compactions to quiesce
    "readseq,"
    "readreverse,"
    "compact,"
    "readrandom,"
    "readseq,"
    "readreverse,"
    "fill100K,"
    "crc32c,"
    "snappycomp,"
    "snappyuncomp,"
    "acquireload,"
    "filter,"
    ;

// Number of key/values to place in database
static int FLAGS_num = 1000000;

// Number of read operations to do.  If negative, do FLAGS_num reads.
static int FLAGS_reads = -1;

// Number of concurrent threads to run.
static int FLAGS_threads = 1;

// Number of concurrent write threads to run.
static int FLAGS_write_threads = 1;

// Number of concurrent read threads to run.
static int FLAGS_read_threads = 1;

// Size of each value
static int FLAGS_value_size = 1024;

// Arrange to generate values that shrink to this fraction of
// their original size after compression
static double FLAGS_compression_ratio = 0.5;

// Print histogram of operation timings
static bool FLAGS_histogram = false;

// Number of bytes to buffer in memtable before compacting
// (initialized to default value by "main")
static int FLAGS_write_buffer_size = 0;

// Number of bytes to use as a cache of uncompressed data.
// Negative means use default settings.
static int FLAGS_cache_size = -1;

// Maximum number of files to keep open at the same time (use default if == 0)
static int FLAGS_open_files = 0;

// Amount of data per block (initialized to default value by "main")
static int FLAGS_block_size = 0;

// Number of next operations to do in a ScanRandom workload
static int FLAGS_num_next = 1;

// Base key which gets added to the randodm key generated
static int FLAGS_base_key = 0;

// Bloom filter bits per key.
// Negative means use default settings.
static int FLAGS_bloom_bits = 10;

// If true, do not destroy the existing database.  If you set this
// flag and also specify a benchmark that wants a fresh database, that
// benchmark will fail.
static bool FLAGS_use_existing_db = false;

// Use the db with the following name.
static const char* FLAGS_db = NULL;

// the parameters of mix_graph
DEFINE_double(keyrange_dist_a, 0.0,
              "The parameter 'a' of prefix average access distribution "
              "f(x)=a*exp(b*x)+c*exp(d*x)");
DEFINE_double(keyrange_dist_b, 0.0,
              "The parameter 'b' of prefix average access distribution "
              "f(x)=a*exp(b*x)+c*exp(d*x)");
DEFINE_double(keyrange_dist_c, 0.0,
              "The parameter 'c' of prefix average access distribution"
              "f(x)=a*exp(b*x)+c*exp(d*x)");
DEFINE_double(keyrange_dist_d, 0.0,
              "The parameter 'd' of prefix average access distribution"
              "f(x)=a*exp(b*x)+c*exp(d*x)");
DEFINE_int64(keyrange_num, 1,
             "The number of key ranges that are in the same prefix "
             "group, each prefix range will have its key access "
             "distribution");
DEFINE_double(key_dist_a, 0.0,
              "The parameter 'a' of key access distribution model "
              "f(x)=a*x^b");
DEFINE_double(key_dist_b, 0.0,
              "The parameter 'b' of key access distribution model "
              "f(x)=a*x^b");
DEFINE_double(value_theta, 0.0,
              "The parameter 'theta' of Generized Pareto Distribution "
              "f(x)=(1/sigma)*(1+k*(x-theta)/sigma)^-(1/k+1)");
DEFINE_double(value_k, 0.0,
              "The parameter 'k' of Generized Pareto Distribution "
              "f(x)=(1/sigma)*(1+k*(x-theta)/sigma)^-(1/k+1)");
DEFINE_double(value_sigma, 0.0,
              "The parameter 'theta' of Generized Pareto Distribution "
              "f(x)=(1/sigma)*(1+k*(x-theta)/sigma)^-(1/k+1)");
DEFINE_double(iter_theta, 0.0,
              "The parameter 'theta' of Generized Pareto Distribution "
              "f(x)=(1/sigma)*(1+k*(x-theta)/sigma)^-(1/k+1)");
DEFINE_double(iter_k, 0.0,
              "The parameter 'k' of Generized Pareto Distribution "
              "f(x)=(1/sigma)*(1+k*(x-theta)/sigma)^-(1/k+1)");
DEFINE_double(iter_sigma, 0.0,
              "The parameter 'sigma' of Generized Pareto Distribution "
              "f(x)=(1/sigma)*(1+k*(x-theta)/sigma)^-(1/k+1)");
DEFINE_double(mix_get_ratio, 1.0,
              "The ratio of Get queries of mix_graph workload");
DEFINE_double(mix_put_ratio, 0.0,
              "The ratio of Put queries of mix_graph workload");
DEFINE_double(mix_seek_ratio, 0.0,
              "The ratio of Seek queries of mix_graph workload");
DEFINE_int64(mix_max_scan_len, 10000, "The max scan length of Iterator");
DEFINE_int64(mix_ave_kv_size, 512,
             "The average key-value size of this workload");
DEFINE_int64(mix_max_value_size, 1024, "The max value size of this workload");
DEFINE_double(
    sine_mix_rate_noise, 0.0,
    "Add the noise ratio to the sine rate, it is between 0.0 and 1.0");
DEFINE_bool(sine_mix_rate, false,
            "Enable the sine QPS control on the mix workload");
DEFINE_uint64(
    sine_mix_rate_interval_milliseconds, 10000,
    "Interval of which the sine wave read_rate_limit is recalculated");
DEFINE_int64(mix_accesses, -1,
             "The total query accesses of mix_graph workload");

DEFINE_uint64(
    benchmark_read_rate_limit, 0,
    "If non-zero, db_bench will rate-limit the reads from RocksDB. This "
    "is the global rate in ops/second.");

DEFINE_int64(seed, 0, "Seed base for random number generators. "
             "When 0 it is deterministic.");

DEFINE_int32(key_size, 16, "size of each key");

DEFINE_int64(keys_per_prefix, 0, "control average number of keys generated "
             "per prefix, 0 means no special handling of the prefix, "
             "i.e. use the prefix comes with the generated random number.");

DEFINE_int32(prefix_size, 0, "control the prefix size for HashSkipList and "
             "plain table");

// The default reduces the overhead of reading time with flash. With HDD, which
// offers much less throughput, however, this number better to be set to 1.
DEFINE_int32(ops_between_duration_checks, 1000,
             "Check duration limit every x ops");

DEFINE_bool(use_existing_keys, false,
            "If true, uses existing keys in the DB, "
            "rather than generating new ones. This involves some startup "
            "latency to load all keys into memory. It is supported for the "
            "same read/overwrite benchmarks as `-use_existing_db=true`, which "
            "must also be set for this flag to be enabled. When this flag is "
            "set, the value for `-num` will be ignored.");

DEFINE_bool(verify_checksum, true,
            "Verify checksum for every block read"
            " from storage");

DEFINE_int32(duration, 0, "Time in seconds for the random-ops tests to run."
             " When 0 then num & reads determine the test duration");

namespace leveldb {

namespace {

// Helper for quickly generating random data.
class RandomGenerator {
 private:
  std::string data_;
  int pos_;

 public:
  RandomGenerator() : data_(), pos_() {
    // We use a limited amount of data over and over again and ensure
    // that it is larger than the compression window (32KB), and also
    // large enough to serve all typical value sizes we want to write.
    Random rnd(301);
    std::string piece;
    while (data_.size() < 1048576) {
      // Add a short fragment that is as compressible as specified
      // by FLAGS_compression_ratio.
      test::CompressibleString(&rnd, FLAGS_compression_ratio, 100, &piece);
      data_.append(piece);
    }
    pos_ = 0;
  }

  Slice Generate(size_t len) {
    if (pos_ + len > data_.size()) {
      pos_ = 0;
      assert(len < data_.size());
    }
    pos_ += len;
    return Slice(data_.data() + pos_ - len, len);
  }
};

static Slice TrimSpace(Slice s) {
  size_t start = 0;
  while (start < s.size() && isspace(s[start])) {
    start++;
  }
  size_t limit = s.size();
  while (limit > start && isspace(s[limit-1])) {
    limit--;
  }
  return Slice(s.data() + start, limit - start);
}

static void AppendWithSpace(std::string* str, Slice msg) {
  if (msg.empty()) return;
  if (!str->empty()) {
    str->push_back(' ');
  }
  str->append(msg.data(), msg.size());
}

class Stats {
 private:
  double start_;
  double finish_;
  double seconds_;
  int done_;
  int next_report_;
  int64_t bytes_;
  double last_op_finish_;
  Histogram hist_;
  std::string message_;

 public:
  Stats() 
    : start_(),
      finish_(),
      seconds_(),
      done_(),
      next_report_(),
      bytes_(),
      last_op_finish_(),
      hist_(),
      message_() {
    Start();
  }

  void Start() {
    next_report_ = 100;
    last_op_finish_ = start_;
    hist_.Clear();
    done_ = 0;
    bytes_ = 0;
    seconds_ = 0;
    start_ = Env::Default()->NowMicros();
    finish_ = start_;
    message_.clear();
  }

  void Merge(const Stats& other) {
    hist_.Merge(other.hist_);
    done_ += other.done_;
    bytes_ += other.bytes_;
    seconds_ += other.seconds_;
    if (other.start_ < start_) start_ = other.start_;
    if (other.finish_ > finish_) finish_ = other.finish_;

    // Just keep the messages from one thread
    if (message_.empty()) message_ = other.message_;
  }

  void Stop() {
    finish_ = Env::Default()->NowMicros();
    seconds_ = (finish_ - start_) * 1e-6;
  }

  void AddMessage(Slice msg) {
    AppendWithSpace(&message_, msg);
  }

  void FinishedSingleOp() {
    if (FLAGS_histogram) {
      double now = Env::Default()->NowMicros();
      double micros = now - last_op_finish_;
      hist_.Add(micros);
      if (micros > 20000) {
        fprintf(stderr, "long op: %.1f micros%30s\r", micros, "");
        fflush(stderr);
      }
      last_op_finish_ = now;
    }

    done_++;
    if (done_ >= next_report_) {
      if      (next_report_ < 1000)   next_report_ += 100;
      else if (next_report_ < 5000)   next_report_ += 500;
      else if (next_report_ < 10000)  next_report_ += 1000;
      else if (next_report_ < 50000)  next_report_ += 5000;
      else if (next_report_ < 100000) next_report_ += 10000;
      else if (next_report_ < 500000) next_report_ += 50000;
      else                            next_report_ += 100000;
      fprintf(stderr, "... finished %d ops%30s\r", done_, "");
      fflush(stderr);
    }
  }

  void AddBytes(int64_t n) {
    bytes_ += n;
  }

  void Report(const Slice& name) {
    // Pretend at least one op was done in case we are running a benchmark
    // that does not call FinishedSingleOp().
    if (done_ < 1) done_ = 1;

    std::string extra;
    if (bytes_ > 0) {
      // Rate is computed on actual elapsed time, not the sum of per-thread
      // elapsed times.
      double elapsed = (finish_ - start_) * 1e-6;
      char rate[100];
      snprintf(rate, sizeof(rate), "%6.1f MB/s",
               (bytes_ / 1048576.0) / elapsed);
      extra = rate;
    }
    AppendWithSpace(&extra, message_);

    fprintf(stdout, "%-12s : %11.3f micros/op;%s%s\n",
            name.ToString().c_str(),
            seconds_ * 1e6 / done_,
            (extra.empty() ? "" : " "),
            extra.c_str());
    if (FLAGS_histogram) {
      fprintf(stdout, "Microseconds per op:\n%s\n", hist_.ToString().c_str());
    }
    fflush(stdout);
  }
};

// State shared by all concurrent executions of the same benchmark.
struct SharedState {
  port::Mutex mu;
  port::CondVar cv;
  int total;

  // Each thread goes through the following states:
  //    (1) initializing
  //    (2) waiting for others to be initialized
  //    (3) running
  //    (4) done

  int num_initialized;
  int num_done;
  bool start;

  SharedState()
    : mu(),
      cv(&mu),
      total(),
      num_initialized(),
      num_done(),
      start() {
  }
};

// Per-thread state for concurrent executions of the same benchmark.
struct ThreadState {
  int tid;             // 0..n-1 when running in n threads
  Random rand;         // Has different seeds for different threads
  Stats stats;
  SharedState* shared;
  Random64 mixgraph_rand;

  ThreadState(int index)
      : tid(index),
        rand(1000 + index),
        mixgraph_rand((FLAGS_seed ? FLAGS_seed : 1000) + index),
        stats(),
        shared() {
  }
 private:
  ThreadState(const ThreadState&);
  ThreadState& operator = (const ThreadState&);
};

}  // namespace


class Benchmark {
 private:
  Benchmark(const Benchmark&);
  Benchmark& operator = (const Benchmark&);
  Cache* cache_;
  const FilterPolicy* filter_policy_;
  DB* db_;
  int num_;
  int value_size_;
  int entries_per_batch_;
  WriteOptions write_options_;
  int reads_;
  int heap_counter_;
  int key_size_;
  double read_random_exp_range_;
  std::vector<std::string> keys_;
  int64_t keys_per_prefix_;
  int prefix_size_;

  DBImpl* dbfull() {
    return reinterpret_cast<DBImpl*>(db_);
  }

  void PrintHeader() {
    const int kKeySize = 16;
    PrintEnvironment();
    fprintf(stdout, "Keys:       %d bytes each\n", kKeySize);
    fprintf(stdout, "Values:     %d bytes each (%d bytes after compression)\n",
            FLAGS_value_size,
            static_cast<int>(FLAGS_value_size * FLAGS_compression_ratio + 0.5));
    fprintf(stdout, "Entries:    %d\n", num_);
    fprintf(stdout, "RawSize:    %.1f MB (estimated)\n",
            ((static_cast<int64_t>(kKeySize + FLAGS_value_size) * num_)
             / 1048576.0));
    fprintf(stdout, "FileSize:   %.1f MB (estimated)\n",
            (((kKeySize + FLAGS_value_size * FLAGS_compression_ratio) * num_)
             / 1048576.0));
    PrintWarnings();
    fprintf(stdout, "------------------------------------------------\n");
  }

  void PrintWarnings() {
#if defined(__GNUC__) && !defined(__OPTIMIZE__)
    fprintf(stdout,
            "WARNING: Optimization is disabled: benchmarks unnecessarily slow\n"
            );
#endif
#ifndef NDEBUG
    fprintf(stdout,
            "WARNING: Assertions are enabled; benchmarks unnecessarily slow\n");
#endif

    // See if snappy is working by attempting to compress a compressible string
    const char text[] = "yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy";
    std::string compressed;
    if (!port::Snappy_Compress(text, sizeof(text), &compressed)) {
      fprintf(stdout, "WARNING: Snappy compression is not enabled\n");
    } else if (compressed.size() >= sizeof(text)) {
      fprintf(stdout, "WARNING: Snappy compression is not effective\n");
    }
  }

  void PrintEnvironment() {
    fprintf(stderr, "LevelDB:    version %d.%d\n",
            kMajorVersion, kMinorVersion);

#if defined(__linux)
    time_t now = time(NULL);
    fprintf(stderr, "Date:       %s", ctime(&now));  // ctime() adds newline

    FILE* cpuinfo = fopen("/proc/cpuinfo", "r");
    if (cpuinfo != NULL) {
      char line[1000];
      int num_cpus = 0;
      std::string cpu_type;
      std::string cache_size;
      while (fgets(line, sizeof(line), cpuinfo) != NULL) {
        const char* sep = strchr(line, ':');
        if (sep == NULL) {
          continue;
        }
        Slice key = TrimSpace(Slice(line, sep - 1 - line));
        Slice val = TrimSpace(Slice(sep + 1));
        if (key == "model name") {
          ++num_cpus;
          cpu_type = val.ToString();
        } else if (key == "cache size") {
          cache_size = val.ToString();
        }
      }
      fclose(cpuinfo);
      fprintf(stderr, "CPU:        %d * %s\n", num_cpus, cpu_type.c_str());
      fprintf(stderr, "CPUCache:   %s\n", cache_size.c_str());
    }
#endif
  }

 public:
  Benchmark()
  : cache_(FLAGS_cache_size >= 0 ? NewLRUCache(FLAGS_cache_size) : NULL),
    filter_policy_(FLAGS_bloom_bits >= 0
                   ? NewBloomFilterPolicy(FLAGS_bloom_bits)
                   : NULL),
    db_(NULL),
    num_(FLAGS_num),
    value_size_(FLAGS_value_size),
    entries_per_batch_(1),
    write_options_(),
    key_size_(FLAGS_key_size),
    read_random_exp_range_(0.0),
    reads_(FLAGS_reads < 0 ? FLAGS_num : FLAGS_reads),
    keys_per_prefix_(FLAGS_keys_per_prefix),
    prefix_size_(FLAGS_prefix_size),
    heap_counter_(0) {
    std::vector<std::string> files;
    Env::Default()->GetChildren(FLAGS_db, &files);
    for (size_t i = 0; i < files.size(); i++) {
      if (Slice(files[i]).starts_with("heap-")) {
        Env::Default()->DeleteFile(std::string(FLAGS_db) + "/" + files[i]);
      }
    }
    if (!FLAGS_use_existing_db) {
      DestroyDB(FLAGS_db, Options());
    }
  }

  ~Benchmark() {
    delete db_;
    delete cache_;
    delete filter_policy_;
  }

  void TryReopen() {
	if (db_ != NULL) {
		delete db_;
	}
    db_ = NULL;
    Open();
  }

  struct trace_operation_t {
  	char cmd;
  	unsigned long long key;
  	unsigned long param;
  };
  struct trace_operation_t *trace_ops[10]; // Assuming maximum of 10 concurrent threads

  struct result_t {
  	unsigned long long ycsbdata;
  	unsigned long long kvdata;
  	unsigned long long ycsb_r;
  	unsigned long long ycsb_d;
  	unsigned long long ycsb_i;
  	unsigned long long ycsb_u;
  	unsigned long long ycsb_s;
  	unsigned long long kv_p;
  	unsigned long long kv_g;
  	unsigned long long kv_d;
  	unsigned long long kv_itseek;
  	unsigned long long kv_itnext;
  };

  struct result_t results[10];

  unsigned long long print_splitup(int tid) {
	struct result_t& result = results[tid];
  	printf("YCSB splitup: R = %llu, D = %llu, I = %llu, U = %llu, S = %llu\n",
  			result.ycsb_r,
  			result.ycsb_d,
  			result.ycsb_i,
  			result.ycsb_u,
  			result.ycsb_s);
  	printf("LevelDB/WiscKey splitup: P = %llu, G = %llu, D = %llu, ItSeek = %llu, ItNext = %llu\n",
  			result.kv_p,
  			result.kv_g,
  			result.kv_d,
  			result.kv_itseek,
  			result.kv_itnext);
  	return result.ycsb_r + result.ycsb_d + result.ycsb_i + result.ycsb_u + result.ycsb_s;
  }

  int split_file_names(const char *file, char file_names[20][100]) {
	  char delimiter = ',';
	  int index  = 0;
	  int cur = 0;
	  for (int i = 0; i < strlen(file); i++) {
		  if (file[i] == ',') {
			  if (cur > 0) {
				  file_names[index][cur] = '\0';
				  index++;
				  cur = 0;
			  }
			  continue;
		  }
		  if (file[i] == ' ') {
			  continue;
		  }
		  file_names[index][cur] = file[i];
		  cur++;
	  }
	  if (cur > 0) {
		  file_names[index][cur] = '\0';
		  cur = 0;
		  index++;
	  }
	  return index;
  }

  void parse_trace(const char *file, int tid) {
  	int ret;
  	char *buf;
  	FILE *fp;
  	size_t bufsize = 1000;
  	struct trace_operation_t *curop = NULL;
  	unsigned long long total_ops = 0;

  	char file_names[20][100];
  	int num_trace_files = split_file_names(file, file_names);

  	const char* corresponding_file;
  	if (tid >= num_trace_files) {
  		corresponding_file = file_names[num_trace_files-1]; // Take the last file if number of files is lesser
  	} else {
  		corresponding_file = file_names[tid];
  	}
  	printf("Thread %d: Parsing trace ...\n", tid);
  	trace_ops[tid] = (struct trace_operation_t *) mmap(NULL, MAX_TRACE_OPS * sizeof(struct trace_operation_t),
  			PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  	if (trace_ops[tid] == MAP_FAILED)
  		perror(NULL);
  	assert(trace_ops[tid] != MAP_FAILED);

  	buf = (char *) malloc(bufsize);
  	assert (buf != NULL);

  	fp = fopen(corresponding_file, "r");
  	assert(fp != NULL);
  	curop = trace_ops[tid];
  	while((ret = getline(&buf, &bufsize, fp)) > 0) {
  		char tmp[1000];
  		ret = sscanf(buf, "%c %llu %lu\n", &curop->cmd, &curop->key, &curop->param);
  		assert(ret == 2 || ret == 3);
  		if (curop->cmd == 'r' || curop->cmd == 'd') {
  			assert(ret == 2);
  			sprintf(tmp, "%c %llu\n", curop->cmd, curop->key);
  			assert(strcmp(tmp, buf) == 0);
  		} else if (curop->cmd == 's' || curop->cmd == 'u' || curop->cmd == 'i') {
  			assert(ret == 3);
  			sprintf(tmp, "%c %llu %lu\n", curop->cmd, curop->key, curop->param);
  			assert(strcmp(tmp, buf) == 0);
  		} else {
  			assert(false);
  		}
  		curop++;
  		total_ops++;
  	}
  	printf("Thread %d: Done parsing, %llu operations.\n", tid, total_ops);
  }

  char valuebuf[MAX_VALUE_SIZE];

  Status perform_op(DB *db, struct trace_operation_t *op, int tid) {
  	char keybuf[100];
  	int keylen;
  	Status status;
  	static struct ReadOptions roptions;
  	static struct WriteOptions woptions;

  	keylen = sprintf(keybuf, "user%llu", op->key);
  	Slice key(keybuf, keylen);

  	struct result_t& result = results[tid];
  	if (op->cmd == 'r') {
  		std::string value;
  		status = db->Get(roptions, key, &value);
  		sassert(status.ok());
  		result.ycsbdata += keylen + value.length();
  		result.kvdata += keylen + value.length();
  		//assert(value.length() == 1080);
  		result.ycsb_r++;
  		result.kv_g++;
  	} else if (op->cmd == 'd') {
  		status = db->Delete(woptions, key);
  		sassert(status.ok());
  		result.ycsbdata += keylen;
  		result.kvdata += keylen;
  		result.ycsb_d++;
  		result.kv_d++;
  	} else if (op->cmd == 'i') {
  		// op->param refers to the size of the value.
  		status = db->Put(woptions, key, Slice(valuebuf, op->param));
  		sassert(status.ok());
  		result.ycsbdata += keylen + op->param;
  		result.kvdata += keylen + op->param;
  		result.ycsb_i++;
  		result.kv_p++;
  	} else if (op->cmd == 'u') {
  		int update_value_size = 1024;
  		status = db->Put(woptions, key, Slice(valuebuf, update_value_size));
  		sassert(status.ok());
  		result.ycsbdata += keylen + op->param;
  		result.kvdata += keylen + update_value_size;
  		result.ycsb_u++;
  		result.kv_g++;
  		result.kv_p++;
  	} else if (op->cmd == 's') {
  		// op->param refers to the number of records to scan.
  		int retrieved = 0;
  		result.kv_itseek++;
  		Iterator *it;
  		it = db->NewIterator(ReadOptions());
  		int range_size = op->param;
  		for (it->Seek(key); it->Valid() && retrieved < range_size; it->Next()) {
  			if (!it->status().ok())
  				std::cerr << "\n\n" << it->status().ToString() << "\n\n";
  			assert(it->status().ok());

  			// Actually retrieving the key and the value, since
  			// that might incur disk reads.
  			unsigned long retvlen = it->value().ToString().length();
  			unsigned long retklen = it->key().ToString().length();
  			result.ycsbdata += retklen + retvlen;
  			result.kvdata += retklen + retvlen;

  			result.kv_itnext++;
  			retrieved ++;
  		}
  		delete it;
  		result.ycsb_s++;
  	} else {
  		assert(false);
  	}
  	return status;
  }

  #define envinput(var, type) {assert(getenv(#var)); int ret = sscanf(getenv(#var), type, &var); assert(ret == 1);}
  #define envstrinput(var) strcpy(var, getenv(#var))

  void YCSB(ThreadState* thread) {
	int tid = thread->tid;
	char trace_file[1000];

	envstrinput(trace_file);

	parse_trace(trace_file, tid);

	struct rlimit rlim;
	rlim.rlim_cur = 1000000;
	rlim.rlim_max = 1000000;
	int ret;// = setrlimit(RLIMIT_NOFILE, &rlim);
//	assert(ret == 0);

	struct trace_operation_t *curop = trace_ops[tid];
	unsigned long long total_ops = 0;
	struct timeval start, end;

	printf("Thread %d: Replaying trace ...\n", tid);

	gettimeofday(&start, NULL);
	fprintf(stderr, "\nCompleted 0 ops");
	fflush(stderr);
	uint64_t succeeded = 0;
	while(curop->cmd) {
		Status status = perform_op(db_, curop, tid);
		if (status.ok()) {
			succeeded++;
		}
		thread->stats.FinishedSingleOp();
		curop++;
		total_ops++;
	}
	PrintStats("leveldb.stats");
	fprintf(stderr, "\r");
	ret = gettimeofday(&end, NULL);
	double secs = (end.tv_sec - start.tv_sec) + double(end.tv_usec - start.tv_usec) / 1000000;

	struct result_t& result = results[tid];
	printf("\n\nThread %d: Done replaying %llu operations.\n", tid, total_ops);
	unsigned long long splitup_ops = print_splitup(tid);
	assert(splitup_ops == total_ops);
	printf("Thread %d: %lu of %llu operations succeeded.\n", tid, succeeded, total_ops);
	printf("Thread %d: Time taken = %0.3lf seconds\n", tid, secs);
	printf("Thread %d: Total data: YCSB = %0.6lf GB, HyperLevelDB = %0.6lf GB\n", tid,
			double(result.ycsbdata) / 1024.0 / 1024.0 / 1024.0,
			double(result.kvdata) / 1024.0 / 1024.0 / 1024.0);
	printf("Thread %d: Ops/s = %0.3lf Kops/s\n", tid, double(total_ops) / 1024.0 / secs);

	double throughput = double(result.ycsbdata) / secs;
	printf("Thread %d: YCSB throughput = %0.6lf MB/s\n", tid, throughput / 1024.0 / 1024.0);
	throughput = double(result.kvdata) / secs;
	printf("Thread %d: HyperLevelDB throughput = %0.6lf MB/s\n", tid, throughput / 1024.0 / 1024.0);
  }

  void ReliabilityCheck(ThreadState* thread) {
	printf("Starting Reliability check to verify if the keys inserted are present in the database . . .\n");
	char file_name[100];
	envstrinput(file_name);
	printf("file_name to read from: %s\n", file_name);

	ReadOptions options;
	std::string value;
	std::ifstream infile;
	infile.open(file_name, std::ios::in);

	char key[100];
	int found = 0, total = 0;
	while (infile >> key) {
		if (db_->Get(options, key, &value).ok()) {
			found++;
		} else {
			printf("ERROR !! Key %s is not found in the database !\n", key);
		}
		total++;
	}
	printf("%d of %d values found in database. \n", found, total);
  }

  void ReliabilityStart(ThreadState* thread) {
	printf("Starting to insert values in random order . . .\n");
	char file_name[100];
	envstrinput(file_name);
	printf("file_name to write to: %s", file_name);

	if (num_ != FLAGS_num) {
	  char msg[100];
	  snprintf(msg, sizeof(msg), "(%d ops)", num_);
	  thread->stats.AddMessage(msg);
	}

	std::ofstream outfile;
	if (FLAGS_use_existing_db) {
		outfile.open(file_name, std::ios::app);
	} else {
		outfile.open(file_name, std::ios::out);
	}

	RandomGenerator gen;
	WriteBatch batch;
	Status s;
	int64_t bytes = 0;

	for (int i = 0; i < num_; i += entries_per_batch_) {
	  batch.Clear();
	  char key[100];

	  for (int j = 0; j < entries_per_batch_; j++) {
		const int k = (thread->rand.Next() % FLAGS_num);
		snprintf(key, sizeof(key), "%016d", k);
		batch.Put(key, gen.Generate(value_size_));
		bytes += value_size_ + strlen(key);
		thread->stats.FinishedSingleOp();
	  }
	  s = db_->Write(write_options_, &batch);
	  outfile << key << std::endl;
	  if (!s.ok()) {
		fprintf(stderr, "put error: %s\n", s.ToString().c_str());
		exit(1);
	  }
	}
	outfile.close();
	thread->stats.AddBytes(bytes);
  }

  void Run() {
    PrintHeader();
    Open();

    const char* benchmarks = FLAGS_benchmarks;
    int num_write_threads = FLAGS_write_threads;
    int num_read_threads = FLAGS_read_threads;
    int num_threads = FLAGS_threads;

    while (benchmarks != NULL) {
      const char* sep = strchr(benchmarks, ',');
      Slice name;
      if (sep == NULL) {
        name = benchmarks;
        benchmarks = NULL;
      } else {
        name = Slice(benchmarks, sep - benchmarks);
        benchmarks = sep + 1;
      }

      // Reset parameters that may be overriddden bwlow
      num_ = FLAGS_num;
      reads_ = (FLAGS_reads < 0 ? FLAGS_num : FLAGS_reads);
      value_size_ = FLAGS_value_size;
      entries_per_batch_ = 1;
      write_options_ = WriteOptions();

      void (Benchmark::*method)(ThreadState*) = NULL;
      bool fresh_db = false;
      int num_threads = FLAGS_threads;

      if (name ==  Slice("ycsb")) {
    	  method = &Benchmark::YCSB;
      } else if (name == Slice("fillseq")) {
        fresh_db = true;
        method = &Benchmark::WriteSeq;
      } else if (name == Slice("fillbatch")) {
        fresh_db = true;
        entries_per_batch_ = 1000;
        method = &Benchmark::WriteSeq;
      } else if (name == Slice("rel_start")) {
//        fresh_db = true;
//        entries_per_batch_ = 1000;
        method = &Benchmark::ReliabilityStart;
      } else if (name == Slice("rel_check")) {
//        fresh_db = true;
//        entries_per_batch_ = 1000;
        method = &Benchmark::ReliabilityCheck;
      } else if (name == Slice("fillrandom")) {
        fresh_db = true;
        method = &Benchmark::WriteRandom;
      } else if (name == Slice("reopen")) {
        fresh_db = false;
        method = &Benchmark::Reopen;
      } else if (name == Slice("overwrite")) {
        fresh_db = false;
        method = &Benchmark::WriteRandom;
      } else if (name == Slice("fillsync")) {
        fresh_db = true;
        num_ /= 1000;
        write_options_.sync = true;
        method = &Benchmark::WriteRandom;
      } else if (name == Slice("fill100K")) {
        fresh_db = true;
        num_ /= 1000;
        value_size_ = 100 * 1000;
        method = &Benchmark::WriteRandom;
      } else if (name == Slice("readseq")) {
        method = &Benchmark::ReadSequential;
      } else if (name == Slice("readreverse")) {
        method = &Benchmark::ReadReverse;
      } else if (name == Slice("readrandom")) {
        method = &Benchmark::ReadRandom;
      } else if (name == Slice("readmissing")) {
        method = &Benchmark::ReadMissing;
      } else if (name == Slice("seekrandom")) {
        method = &Benchmark::SeekRandom;
      } else if (name == Slice("scanrandom")) {
        method = &Benchmark::ScanRandom;
      } else if (name == Slice("readhot")) {
        method = &Benchmark::ReadHot;
      } else if (name == Slice("readrandomsmall")) {
        reads_ /= 1000;
        method = &Benchmark::ReadRandom;
      } else if (name == Slice("deleteseq")) {
        method = &Benchmark::DeleteSeq;
      } else if (name == Slice("deleterandom")) {
        method = &Benchmark::DeleteRandom;
      } else if (name == Slice("readwhilewriting")) {
        num_threads++;  // Add extra thread for writing
        fresh_db = false;
        method = &Benchmark::ReadWhileWriting;
      } else if (name == Slice("seekwhilewriting")) {
        num_threads++;  // Add extra thread for writing
        method = &Benchmark::SeekWhileWriting;
      } else if (name == Slice("compact")) {
        method = &Benchmark::Compact;
      } else if (name == Slice("crc32c")) {
        method = &Benchmark::Crc32c;
      } else if (name == Slice("acquireload")) {
        method = &Benchmark::AcquireLoad;
      } else if (name == Slice("snappycomp")) {
        method = &Benchmark::SnappyCompress;
      } else if (name == Slice("snappyuncomp")) {
        method = &Benchmark::SnappyUncompress;
      } else if (name == Slice("heapprofile")) {
        HeapProfile();
      } else if (name == Slice("stats")) {
        PrintStats("leveldb.stats");
      } else if (name == Slice("sstables")) {
        PrintStats("leveldb.sstables");
      } else if (name == Slice("compactsinglelevel")) {
    	fresh_db = false;
    	method = &Benchmark::WaitForStableStateSinglLevel;
      } else if (name == Slice("compactalllevels")) {
    	fresh_db = false;
    	method = &Benchmark::CompactAllLevels;
      } else if (name == Slice("compactonce")) {
    	fresh_db = false;
    	method = &Benchmark::CompactOnce;
      } else if (name == Slice("reducelevelsby1")) {
    	fresh_db = false;
    	method = &Benchmark::ReduceActiveLevelsByOne;
      } else if (name == Slice("compactmemtable")) {
    	fresh_db = false;
    	method = &Benchmark::CompactMemtable;
      } else if (name == Slice("printdb")) {
    	fresh_db = false;
    	method = &Benchmark::PrintDB;
      } else if (name == Slice("filter")) {
        PrintStats("leveldb.filter");
      } else if (name == Slice("mixgraph")) {
        fresh_db=false;
        method = &Benchmark::MixGraph;
      }
      else {
        if (name != Slice()) {  // No error message for empty name
          fprintf(stderr, "unknown benchmark '%s'\n", name.ToString().c_str());
        }
      }

      if (fresh_db) {
        if (FLAGS_use_existing_db) {
          fprintf(stdout, "%-12s : skipped (--use_existing_db is true)\n",
                  name.ToString().c_str());
          method = NULL;
        } else {
          delete db_;
          db_ = NULL;
          DestroyDB(FLAGS_db, Options());
          Open();
        }
      }

      if (method != NULL) {
        RunBenchmark(num_threads, name, method);
      }
    }
    db_->PrintTimerAudit();
  }

  void print_current_db_contents() {
	  std::string current_db_state;
	  printf("----------------------Current DB state-----------------------\n");
	  if (db_ == NULL) {
		printf("db_ is NULL !!\n");
		return;
	  }
	  db_->GetCurrentVersionState(&current_db_state);
	  printf("%s\n", current_db_state.c_str());
	  printf("-------------------------------------------------------------\n");
  }

  std::string IterStatus(Iterator* iter) {
    std::string result;
    if (iter->Valid()) {
      result = iter->key().ToString() + "->" + iter->value().ToString();
    } else {
      result = "(invalid)";
    }
    return result;
  }

  int VerifyIteration(int print_every = 100000000) {
	int count = 0;
    std::vector<std::string> forward;
    std::string result;
    Iterator* iter = db_->NewIterator(ReadOptions());
    for (iter->SeekToFirst(); iter->Valid(); iter->Next()) {
      std::string s = IterStatus(iter);
      result.push_back('(');
      result.append(s);
      result.push_back(')');
      forward.push_back(s);
      count++;
      if (count % print_every == 0) {
    	  printf("VerifyIteration :: Seeked %d entries in forward direction.\n", count);
      }
    }

    // Check reverse iteration results are the reverse of forward results
    size_t matched = 0;
    for (iter->SeekToLast(); iter->Valid(); iter->Prev()) {
      ASSERT_LT(matched, forward.size());
      ASSERT_EQ(IterStatus(iter), forward[forward.size() - matched - 1]);
      matched++;
      if (matched % print_every == 0) {
    	  printf("VerifyIteration :: Seeked %lu entries in reverse direction.\n", matched);
      }
    }
    ASSERT_EQ(matched, forward.size());

    delete iter;
    return forward.size();
  }

 private:
  struct ThreadArg {
    Benchmark* bm;
    SharedState* shared;
    ThreadState* thread;
    void (Benchmark::*method)(ThreadState*);
  };

  static void ThreadBody(void* v) {
    ThreadArg* arg = reinterpret_cast<ThreadArg*>(v);
    SharedState* shared = arg->shared;
    ThreadState* thread = arg->thread;
    {
      MutexLock l(&shared->mu);
      shared->num_initialized++;
      if (shared->num_initialized >= shared->total) {
        shared->cv.SignalAll();
      }
      while (!shared->start) {
        shared->cv.Wait();
      }
    }

    thread->stats.Start();
    (arg->bm->*(arg->method))(thread);
    thread->stats.Stop();

    {
      MutexLock l(&shared->mu);
      shared->num_done++;
      if (shared->num_done >= shared->total) {
        shared->cv.SignalAll();
      }
    }
  }

  void RunBenchmark(int n, Slice name,
                    void (Benchmark::*method)(ThreadState*)) {
    SharedState shared;
    shared.total = n;
    shared.num_initialized = 0;
    shared.num_done = 0;
    shared.start = false;

    ThreadArg* arg = new ThreadArg[n];
    for (int i = 0; i < n; i++) {
      arg[i].bm = this;
      arg[i].method = method;
      arg[i].shared = &shared;
      arg[i].thread = new ThreadState(i);
      arg[i].thread->shared = &shared;
      Env::Default()->StartThread(ThreadBody, &arg[i]);
    }

    shared.mu.Lock();
    while (shared.num_initialized < n) {
      shared.cv.Wait();
    }

    shared.start = true;
    shared.cv.SignalAll();
    while (shared.num_done < n) {
      shared.cv.Wait();
    }
    shared.mu.Unlock();

    for (int i = 1; i < n; i++) {
      arg[0].thread->stats.Merge(arg[i].thread->stats);
    }
    arg[0].thread->stats.Report(name);

    for (int i = 0; i < n; i++) {
      delete arg[i].thread;
    }
    delete[] arg;
  }

  void Crc32c(ThreadState* thread) {
    // Checksum about 500MB of data total
    const int size = 4096;
    const char* label = "(4K per op)";
    std::string data(size, 'x');
    int64_t bytes = 0;
    uint32_t crc = 0;
    while (bytes < 500 * 1048576) {
      crc = crc32c::Value(data.data(), size);
      thread->stats.FinishedSingleOp();
      bytes += size;
    }
    // Print so result is not dead
    fprintf(stderr, "... crc=0x%x\r", static_cast<unsigned int>(crc));

    thread->stats.AddBytes(bytes);
    thread->stats.AddMessage(label);
  }

  void AcquireLoad(ThreadState* thread) {
    int dummy;
    port::AtomicPointer ap(&dummy);
    int count = 0;
    void *ptr = NULL;
    thread->stats.AddMessage("(each op is 1000 loads)");
    while (count < 100000) {
      for (int i = 0; i < 1000; i++) {
        ptr = ap.Acquire_Load();
      }
      count++;
      thread->stats.FinishedSingleOp();
    }
    if (ptr == NULL) exit(1); // Disable unused variable warning.
  }

  void SnappyCompress(ThreadState* thread) {
    RandomGenerator gen;
    Slice input = gen.Generate(Options().block_size);
    int64_t bytes = 0;
    int64_t produced = 0;
    bool ok = true;
    std::string compressed;
    while (ok && bytes < 1024 * 1048576) {  // Compress 1G
      ok = port::Snappy_Compress(input.data(), input.size(), &compressed);
      produced += compressed.size();
      bytes += input.size();
      thread->stats.FinishedSingleOp();
    }

    if (!ok) {
      thread->stats.AddMessage("(snappy failure)");
    } else {
      char buf[100];
      snprintf(buf, sizeof(buf), "(output: %.1f%%)",
               (produced * 100.0) / bytes);
      thread->stats.AddMessage(buf);
      thread->stats.AddBytes(bytes);
    }
  }

  void SnappyUncompress(ThreadState* thread) {
    RandomGenerator gen;
    Slice input = gen.Generate(Options().block_size);
    std::string compressed;
    bool ok = port::Snappy_Compress(input.data(), input.size(), &compressed);
    int64_t bytes = 0;
    char* uncompressed = new char[input.size()];
    while (ok && bytes < 1024 * 1048576) {  // Compress 1G
      ok =  port::Snappy_Uncompress(compressed.data(), compressed.size(),
                                    uncompressed);
      bytes += input.size();
      thread->stats.FinishedSingleOp();
    }
    delete[] uncompressed;

    if (!ok) {
      thread->stats.AddMessage("(snappy failure)");
    } else {
      thread->stats.AddBytes(bytes);
    }
  }

  void Open() {
    assert(db_ == NULL);
    Options options;
    options.create_if_missing = !FLAGS_use_existing_db;
    options.block_cache = cache_;
    options.write_buffer_size = FLAGS_write_buffer_size;
    options.max_open_files = FLAGS_open_files;
    options.block_size = FLAGS_block_size;
    options.filter_policy = filter_policy_;
    Status s = DB::Open(options, FLAGS_db, &db_);
    if (!s.ok()) {
      fprintf(stderr, "open error: %s\n", s.ToString().c_str());
      exit(1);
    }
  }

  void PrintDB(ThreadState* thread) {
	  print_current_db_contents();
  }

  void CompactOnce(ThreadState* thread) {
	printf("Compacting the database once . . \n");
	dbfull()->TEST_CompactOnce();
  }

  void ReduceActiveLevelsByOne(ThreadState* thread) {
	printf("Reducing active levels by one . . \n");
	dbfull()->TEST_ReduceNumActiveLevelsByOne();
  }

  void CompactMemtable(ThreadState* thread) {
	printf("Compacting memtable . . \n");
	dbfull()->TEST_CompactMemTable();
  }

  void CompactAllLevels(ThreadState* thread) {
	print_current_db_contents();
	printf("Waiting for all levels to become fully compacted . . ");
	dbfull()->TEST_CompactAllLevels();
	print_current_db_contents();
  }

  void WaitForStableState(ThreadState* thread) {

  }

  void WaitForStableStateSinglLevel(ThreadState* thread) {
	    if (num_ != FLAGS_num) {
	      char msg[100];
	      snprintf(msg, sizeof(msg), "(%d ops)", num_);
	      thread->stats.AddMessage(msg);
	    }

	    print_current_db_contents();
	    printf("Compacting DB to single level . . \n");
	    dbfull()->TEST_ComapactFilesToSingleLevel();
	    printf("After compacting to single level -- ");
	    print_current_db_contents();
  }

  void WriteSeq(ThreadState* thread) {
    DoWrite(thread, true);
  }

  void Reopen(ThreadState* thread) {
	printf("Reopening database . . \n");
	TryReopen();
  }

  void WriteRandom(ThreadState* thread) {
    DoWrite(thread, false);
  }

  // The inverse function of Pareto distribution
  int64_t ParetoCdfInversion(double u, double theta, double k, double sigma) {
    double ret;
    if (k == 0.0) {
      ret = theta - sigma * std::log(u);
    } else {
      ret = theta + sigma * (std::pow(u, -1 * k) - 1) / k;
    }
    return static_cast<int64_t>(ceil(ret));
  }
  // The inverse function of power distribution (y=ax^b)
  int64_t PowerCdfInversion(double u, double a, double b) {
    double ret;
    ret = std::pow((u / a), (1 / b));
    return static_cast<int64_t>(ceil(ret));
  }

  // Add the noice to the QPS
  /*
  double AddNoise(double origin, double noise_ratio) {
    if (noise_ratio < 0.0 || noise_ratio > 1.0) {
      return origin;
    }
    int band_int = static_cast<int>(FLAGS_sine_a);
    double delta = (rand() % band_int - band_int / 2) * noise_ratio;
    if (origin + delta < 0) {
      return origin;
    } else {
      return (origin + delta);
    }
  }
  */

  // Decide the ratio of different query types
  // 0 Get, 1 Put, 2 Seek, 3 SeekForPrev, 4 Delete, 5 SingleDelete, 6 merge
  class QueryDecider {
   public:
    std::vector<int> type_;
    std::vector<double> ratio_;
    int range_;

    QueryDecider() {}
    ~QueryDecider() {}

    Status Initiate(std::vector<double> ratio_input) {
      int range_max = 1000;
      double sum = 0.0;
      for (auto& ratio : ratio_input) {
        sum += ratio;
      }
      range_ = 0;
      for (auto& ratio : ratio_input) {
        range_ += static_cast<int>(ceil(range_max * (ratio / sum)));
        type_.push_back(range_);
        ratio_.push_back(ratio / sum);
      }
      return Status::OK();
    }

    int GetType(int64_t rand_num) {
      if (rand_num < 0) {
        rand_num = rand_num * (-1);
      }
      assert(range_ != 0);
      int pos = static_cast<int>(rand_num % range_);
      for (int i = 0; i < static_cast<int>(type_.size()); i++) {
        if (pos < type_[i]) {
          return i;
        }
      }
      return 0;
    }
  };

  // KeyrangeUnit is the struct of a keyrange. It is used in a keyrange vector
  // to transfer a random value to one keyrange based on the hotness.
  struct KeyrangeUnit {
    int64_t keyrange_start;
    int64_t keyrange_access;
    int64_t keyrange_keys;
  };

  // From our observations, the prefix hotness (key-range hotness) follows
  // the two-term-exponential distribution: f(x) = a*exp(b*x) + c*exp(d*x).
  // However, we cannot directly use the inverse function to decide a
  // key-range from a random distribution. To achieve it, we create a list of
  // KeyrangeUnit, each KeyrangeUnit occupies a range of integers whose size is
  // decided based on the hotness of the key-range. When a random value is
  // generated based on uniform distribution, we map it to the KeyrangeUnit Vec
  // and one KeyrangeUnit is selected. The probability of a  KeyrangeUnit being
  // selected is the same as the hotness of this KeyrangeUnit. After that, the
  // key can be randomly allocated to the key-range of this KeyrangeUnit, or we
  // can based on the power distribution (y=ax^b) to generate the offset of
  // the key in the selected key-range. In this way, we generate the keyID
  // based on the hotness of the prefix and also the key hotness distribution.
  class GenerateTwoTermExpKeys {
   public:
    int64_t keyrange_rand_max_;
    int64_t keyrange_size_;
    int64_t keyrange_num_;
    bool initiated_;
    std::vector<KeyrangeUnit> keyrange_set_;

    GenerateTwoTermExpKeys() {
      keyrange_rand_max_ = FLAGS_num;
      initiated_ = false;
    }

    ~GenerateTwoTermExpKeys() {}

    // Initiate the KeyrangeUnit vector and calculate the size of each
    // KeyrangeUnit.
    Status InitiateExpDistribution(int64_t total_keys, double prefix_a,
                                   double prefix_b, double prefix_c,
                                   double prefix_d) {
      int64_t amplify = 0;
      int64_t keyrange_start = 0;
      initiated_ = true;
      if (FLAGS_keyrange_num <= 0) {
        keyrange_num_ = 1;
      } else {
        keyrange_num_ = FLAGS_keyrange_num;
      }
      keyrange_size_ = total_keys / keyrange_num_;

      // Calculate the key-range shares size based on the input parameters
      for (int64_t pfx = keyrange_num_; pfx >= 1; pfx--) {
        // Step 1. Calculate the probability that this key range will be
        // accessed in a query. It is based on the two-term expoential
        // distribution
        double keyrange_p = prefix_a * std::exp(prefix_b * pfx) +
                            prefix_c * std::exp(prefix_d * pfx);
        if (keyrange_p < std::pow(10.0, -16.0)) {
          keyrange_p = 0.0;
        }
        // Step 2. Calculate the amplify
        // In order to allocate a query to a key-range based on the random
        // number generated for this query, we need to extend the probability
        // of each key range from [0,1] to [0, amplify]. Amplify is calculated
        // by 1/(smallest key-range probability). In this way, we ensure that
        // all key-ranges are assigned with an Integer that  >=0
        if (amplify == 0 && keyrange_p > 0) {
          amplify = static_cast<int64_t>(std::floor(1 / keyrange_p)) + 1;
        }

        // Step 3. For each key-range, we calculate its position in the
        // [0, amplify] range, including the start, the size (keyrange_access)
        KeyrangeUnit p_unit;
        p_unit.keyrange_start = keyrange_start;
        if (0.0 >= keyrange_p) {
          p_unit.keyrange_access = 0;
        } else {
          p_unit.keyrange_access =
              static_cast<int64_t>(std::floor(amplify * keyrange_p));
        }
        p_unit.keyrange_keys = keyrange_size_;
        keyrange_set_.push_back(p_unit);
        keyrange_start += p_unit.keyrange_access;
      }
      keyrange_rand_max_ = keyrange_start;

      // Step 4. Shuffle the key-ranges randomly
      // Since the access probability is calculated from small to large,
      // If we do not re-allocate them, hot key-ranges are always at the end
      // and cold key-ranges are at the begin of the key space. Therefore, the
      // key-ranges are shuffled and the rand seed is only decide by the
      // key-range hotness distribution. With the same distribution parameters
      // the shuffle results are the same.
      Random64 rand_loca(keyrange_rand_max_);
      for (int64_t i = 0; i < FLAGS_keyrange_num; i++) {
        int64_t pos = rand_loca.Next() % FLAGS_keyrange_num;
        assert(i >= 0 && i < static_cast<int64_t>(keyrange_set_.size()) &&
               pos >= 0 && pos < static_cast<int64_t>(keyrange_set_.size()));
        std::swap(keyrange_set_[i], keyrange_set_[pos]);
      }

      // Step 5. Recalculate the prefix start postion after shuffling
      int64_t offset = 0;
      for (auto& p_unit : keyrange_set_) {
        p_unit.keyrange_start = offset;
        offset += p_unit.keyrange_access;
      }

      return Status::OK();
    }

    // Generate the Key ID according to the input ini_rand and key distribution
    int64_t DistGetKeyID(int64_t ini_rand, double key_dist_a,
                         double key_dist_b) {
      int64_t keyrange_rand = ini_rand % keyrange_rand_max_;

      // Calculate and select one key-range that contains the new key
      int64_t start = 0, end = static_cast<int64_t>(keyrange_set_.size());
      while (start + 1 < end) {
        int64_t mid = start + (end - start) / 2;
        assert(mid >= 0 && mid < static_cast<int64_t>(keyrange_set_.size()));
        if (keyrange_rand < keyrange_set_[mid].keyrange_start) {
          end = mid;
        } else {
          start = mid;
        }
      }
      int64_t keyrange_id = start;

      // Select one key in the key-range and compose the keyID
      int64_t key_offset = 0, key_seed;
      if (key_dist_a == 0.0 || key_dist_b == 0.0) {
        key_offset = ini_rand % keyrange_size_;
      } else {
        double u =
            static_cast<double>(ini_rand % keyrange_size_) / keyrange_size_;
        key_seed = static_cast<int64_t>(
            ceil(std::pow((u / key_dist_a), (1 / key_dist_b))));
        Random64 rand_key(key_seed);
        key_offset = rand_key.Next() % keyrange_size_;
      }
      return keyrange_size_ * keyrange_id + key_offset;
    }
  };

  Slice AllocateKey(std::unique_ptr<const char[]>* key_guard) {
    char* data = new char[key_size_];
    const char* const_data = data;
    key_guard->reset(const_data);
    return Slice(key_guard->get(), key_size_);
  }

  class Duration {
 public:
  Duration(uint64_t max_seconds, int64_t max_ops, int64_t ops_per_stage = 0) {
    max_seconds_ = max_seconds;
    max_ops_= max_ops;
    ops_per_stage_ = (ops_per_stage > 0) ? ops_per_stage : max_ops;
    ops_ = 0;
    start_at_ = Env::Default()->NowMicros();
  }

  int64_t GetStage() { return std::min(ops_, max_ops_ - 1) / ops_per_stage_; }

  bool Done(int64_t increment) {
    if (increment <= 0) increment = 1;    // avoid Done(0) and infinite loops
    ops_ += increment;

    if (max_seconds_) {
      // Recheck every appx 1000 ops (exact iff increment is factor of 1000)
      auto granularity = FLAGS_ops_between_duration_checks;
      if ((ops_ / granularity) != ((ops_ - increment) / granularity)) {
        uint64_t now = Env::Default()->NowMicros();
        return ((now - start_at_) / 1000000) >= max_seconds_;
      } else {
        return false;
      }
    } else {
      return ops_ > max_ops_;
    }
  }

 private:
  uint64_t max_seconds_;
  int64_t max_ops_;
  int64_t ops_per_stage_;
  int64_t ops_;
  uint64_t start_at_;
};

  int64_t GetRandomKey(Random64* rand) {
    uint64_t rand_int = rand->Next();
    int64_t key_rand;
    if (read_random_exp_range_ == 0) {
      key_rand = rand_int % FLAGS_num;
    } else {
      const uint64_t kBigInt = static_cast<uint64_t>(1U) << 62;
      long double order = -static_cast<long double>(rand_int % kBigInt) /
                          static_cast<long double>(kBigInt) *
                          read_random_exp_range_;
      long double exp_ran = std::exp(order);
      uint64_t rand_num =
          static_cast<int64_t>(exp_ran * static_cast<long double>(FLAGS_num));
      // Map to a different number to avoid locality.
      const uint64_t kBigPrime = 0x5bd1e995;
      // Overflow is like %(2^64). Will have little impact of results.
      key_rand = static_cast<int64_t>((rand_num * kBigPrime) % FLAGS_num);
    }
    return key_rand;
  }

  // Generate key according to the given specification and random number.
  // The resulting key will have the following format (if keys_per_prefix_
  // is positive), extra trailing bytes are either cut off or padded with '0'.
  // The prefix value is derived from key value.
  //   ----------------------------
  //   | prefix 00000 | key 00000 |
  //   ----------------------------
  // If keys_per_prefix_ is 0, the key is simply a binary representation of
  // random number followed by trailing '0's
  //   ----------------------------
  //   |        key 00000         |
  //   ----------------------------
  void GenerateKeyFromInt(uint64_t v, int64_t num_keys, Slice* key) {
    if (!keys_.empty()) {
      assert(FLAGS_use_existing_keys);
      assert(keys_.size() == static_cast<size_t>(num_keys));
      assert(v < static_cast<uint64_t>(num_keys));
      *key = keys_[v];
      return;
    }
    char* start = const_cast<char*>(key->data());
    char* pos = start;
    if (keys_per_prefix_ > 0) {
      int64_t num_prefix = num_keys / keys_per_prefix_;
      int64_t prefix = v % num_prefix;
      int bytes_to_fill = std::min(prefix_size_, 8);
      if (port::kLittleEndian) {
        for (int i = 0; i < bytes_to_fill; ++i) {
          pos[i] = (prefix >> ((bytes_to_fill - i - 1) << 3)) & 0xFF;
        }
      } else {
        memcpy(pos, static_cast<void*>(&prefix), bytes_to_fill);
      }
      if (prefix_size_ > 8) {
        // fill the rest with 0s
        memset(pos + 8, '0', prefix_size_ - 8);
      }
      pos += prefix_size_;
    }

    int bytes_to_fill = std::min(key_size_ - static_cast<int>(pos - start), 8);
    if (port::kLittleEndian) {
      for (int i = 0; i < bytes_to_fill; ++i) {
        pos[i] = (v >> ((bytes_to_fill - i - 1) << 3)) & 0xFF;
      }
    } else {
      memcpy(pos, static_cast<void*>(&v), bytes_to_fill);
    }
    pos += bytes_to_fill;
    if (key_size_ > pos - start) {
      memset(pos, '0', key_size_ - (pos - start));
    }
  }

  // The social graph wokrload mixed with Get, Put, Iterator queries.
  // The value size and iterator length follow Pareto distribution.
  // The overall key access follow power distribution. If user models the
  // workload based on different key-ranges (or different prefixes), user
  // can use two-term-exponential distribution to fit the workload. User
  // needs to decides the ratio between Get, Put, Iterator queries before
  // starting the benchmark.
  void MixGraph(ThreadState* thread) {
    int64_t read = 0;  // including single gets and Next of iterators
    int64_t gets = 0;
    int64_t puts = 0;
    int64_t found = 0;
    int64_t seek = 0;
    int64_t seek_found = 0;
    int64_t bytes = 0;
    const int64_t default_value_max = 1 * 1024 * 1024;
    int64_t value_max = default_value_max;
    int64_t scan_len_max = FLAGS_mix_max_scan_len;
    double write_rate = 1000000.0;
    double read_rate = 1000000.0;
    bool use_prefix_modeling = false;
    bool use_random_modeling = false;
    GenerateTwoTermExpKeys gen_exp;
    std::vector<double> ratio{FLAGS_mix_get_ratio, FLAGS_mix_put_ratio,
                              FLAGS_mix_seek_ratio};
    char value_buffer[default_value_max];
    QueryDecider query;
    RandomGenerator gen;
    Status s;

    //young:: add
    std::string value;

    if (value_max > FLAGS_mix_max_value_size) {
      value_max = FLAGS_mix_max_value_size;
    }

    //ReadOptions options(FLAGS_verify_checksum, true);
    ReadOptions options;
    std::unique_ptr<const char[]> key_guard;
    Slice key = AllocateKey(&key_guard);
    //PinnableSlice pinnable_val;
    query.Initiate(ratio);

    // the limit of qps initiation
    //if (FLAGS_sine_a != 0 || FLAGS_sine_d != 0) {
    //  thread->shared->read_rate_limiter.reset(NewGenericRateLimiter(
    //      static_cast<int64_t>(read_rate), 100000 /* refill_period_us */, 10 /* fairness */,
    //      RateLimiter::Mode::kReadsOnly));
    // thread->shared->write_rate_limiter.reset(
    //      NewGenericRateLimiter(static_cast<int64_t>(write_rate)));
    //}

    // Decide if user wants to use prefix based key generation
    if (FLAGS_keyrange_dist_a != 0.0 || FLAGS_keyrange_dist_b != 0.0 ||
        FLAGS_keyrange_dist_c != 0.0 || FLAGS_keyrange_dist_d != 0.0) {
      use_prefix_modeling = true;
      gen_exp.InitiateExpDistribution(
          FLAGS_num, FLAGS_keyrange_dist_a, FLAGS_keyrange_dist_b,
          FLAGS_keyrange_dist_c, FLAGS_keyrange_dist_d);
    }
    if (FLAGS_key_dist_a == 0 || FLAGS_key_dist_b == 0) {
      use_random_modeling = true;
    }

    Duration duration(FLAGS_duration, reads_);
    while (!duration.Done(1)) {
      // DBWithColumnFamilies* db_with_cfh = SelectDBWithCfh(thread);
      int64_t ini_rand, rand_v, key_rand, key_seed;
      ini_rand = GetRandomKey(&thread->mixgraph_rand);
      rand_v = ini_rand % FLAGS_num;
      double u = static_cast<double>(rand_v) / FLAGS_num;

      // Generate the keyID based on the key hotness and prefix hotness
      if (use_random_modeling) {
        key_rand = ini_rand;
      } else if (use_prefix_modeling) {
        key_rand =
            gen_exp.DistGetKeyID(ini_rand, FLAGS_key_dist_a, FLAGS_key_dist_b);
      } else {
        key_seed = PowerCdfInversion(u, FLAGS_key_dist_a, FLAGS_key_dist_b);
        Random64 rand(key_seed);
        key_rand = static_cast<int64_t>(rand.Next()) % FLAGS_num;
      }
      GenerateKeyFromInt(key_rand, FLAGS_num, &key);
      int query_type = query.GetType(rand_v);
      
      /*
      // change the qps
      uint64_t now = FLAGS_env->NowMicros();
      uint64_t usecs_since_last;
      if (now > thread->stats.GetSineInterval()) {
        usecs_since_last = now - thread->stats.GetSineInterval();
      } else {
        usecs_since_last = 0;
      }
      */

      /*
      if (usecs_since_last >
          (FLAGS_sine_mix_rate_interval_milliseconds * uint64_t{1000})) {
        double usecs_since_start =
            static_cast<double>(now - thread->stats.GetStart());
        thread->stats.ResetSineInterval();
        double mix_rate_with_noise = AddNoise(
            SineRate(usecs_since_start / 1000000.0), FLAGS_sine_mix_rate_noise);
        read_rate = mix_rate_with_noise * (query.ratio_[0] + query.ratio_[2]);
        write_rate =
            mix_rate_with_noise * query.ratio_[1] * FLAGS_mix_ave_kv_size;

        thread->shared->write_rate_limiter.reset(
            NewGenericRateLimiter(static_cast<int64_t>(write_rate)));
        thread->shared->read_rate_limiter.reset(NewGenericRateLimiter(
            static_cast<int64_t>(read_rate),
            FLAGS_sine_mix_rate_interval_milliseconds * uint64_t{1000}, 10,
            RateLimiter::Mode::kReadsOnly));
      }
      */

      // Start the query
      if (query_type == 0) {
        // the Get query
        gets++;
        read++;
        
        value.clear();
        s = db_->Get(options, key, &value);

        if (s.ok()) {
          found++;
          bytes += key.size() + value.size();
        } else if (!s.IsNotFound()) {
          fprintf(stderr, "Get returned an error: %s\n", s.ToString().c_str());
          abort();
        }

        //if (thread->shared->read_rate_limiter.get() != nullptr &&
        //    read % 256 == 255) {
        //  thread->shared->read_rate_limiter->Request(
        //      256, Env::IO_HIGH, nullptr /* stats */,
        //      RateLimiter::OpType::kRead);
        //}

        //thread->stats.FinishedOps(db_with_cfh, db_with_cfh->db, 1, kRead);
        thread->stats.FinishedSingleOp();
      } else if (query_type == 1) {
        // the Put query
        puts++;
        int64_t val_size = ParetoCdfInversion(
            u, FLAGS_value_theta, FLAGS_value_k, FLAGS_value_sigma);
        if (val_size < 0) {
          val_size = 10;
        } else if (val_size > value_max) {
          val_size = val_size % value_max;
        }
        s = db_->Put(
            write_options_, key,
            gen.Generate(static_cast<unsigned int>(val_size)));
        if (!s.ok()) {
          fprintf(stderr, "put error: %s\n", s.ToString().c_str());
          exit(1);
        }

        
        //if (thread->shared->write_rate_limiter) {
        //  thread->shared->write_rate_limiter->Request(
        //      key.size() + val_size, Env::IO_HIGH, nullptr /*stats*/,
        //      RateLimiter::OpType::kWrite);
        //}
        
        //thread->stats.FinishedOps(db_with_cfh, db_with_cfh->db, 1, kWrite);
        thread->stats.FinishedSingleOp();
      } else if (query_type == 2) {
        // Seek query
        if (db_ != nullptr) {
          Iterator* single_iter = nullptr;
          single_iter = db_->NewIterator(options);
          if (single_iter != nullptr) {
            single_iter->Seek(key);
            seek++;
            read++;
            if (single_iter->Valid() && single_iter->key().compare(key) == 0) {
              seek_found++;
            }
            int64_t scan_length =
                ParetoCdfInversion(u, FLAGS_iter_theta, FLAGS_iter_k,
                                   FLAGS_iter_sigma) %
                scan_len_max;
            for (int64_t j = 0; j < scan_length && single_iter->Valid(); j++) {
              Slice value = single_iter->value();
              memcpy(value_buffer, value.data(),
                     std::min(value.size(), sizeof(value_buffer)));
              bytes += single_iter->key().size() + single_iter->value().size();
              single_iter->Next();
              assert(single_iter->status().ok());
            }
          }
          delete single_iter;
        }
        //thread->stats.FinishedOps(db_with_cfh, db_with_cfh->db, 1, kSeek);
        thread->stats.FinishedSingleOp();
      }
    }

    char msg[256];
    snprintf(msg, sizeof(msg),
             "( Gets:%" PRIu64 " Puts:%" PRIu64 " Seek:%" PRIu64 " of %" PRIu64
             " in %" PRIu64 " found)\n",
             gets, puts, seek, found, read);

    thread->stats.AddBytes(bytes);
    thread->stats.AddMessage(msg);
  
    //if (FLAGS_perf_level > ROCKSDB_NAMESPACE::PerfLevel::kDisable) {
    //  thread->stats.AddMessage(std::string("PERF_CONTEXT:\n") +
    //                           get_perf_context()->ToString());
    //}
  }

  void DoWrite(ThreadState* thread, bool seq) {
	uint64_t before, after, before_g, after_g;
    if (num_ != FLAGS_num) {
      char msg[100];
      snprintf(msg, sizeof(msg), "(%d ops)", num_);
      thread->stats.AddMessage(msg);
    }

    RandomGenerator gen;
    WriteBatch batch;
    Status s;
    int64_t bytes = 0;

    micros(before_g);
    for (int i = 0; i < num_; i += entries_per_batch_) {
      batch.Clear();
      for (int j = 0; j < entries_per_batch_; j++) {
        const int k = seq ? i+j + FLAGS_base_key : (thread->rand.Next() % FLAGS_num) + FLAGS_base_key;
        char key[100];
        snprintf(key, sizeof(key), "%016d", k);
        batch.Put(key, gen.Generate(value_size_));
        bytes += value_size_ + strlen(key);
        thread->stats.FinishedSingleOp();
      }
      s = db_->Write(write_options_, &batch);
      if (!s.ok()) {
        fprintf(stderr, "put error: %s\n", s.ToString().c_str());
        exit(1);
      }
    }
    micros(after_g);
    print_timer_info("DoWrite() method :: Total time took to insert all entries", after_g, before_g);
    thread->stats.AddBytes(bytes);
  }

  void ReadSequential(ThreadState* thread) {
    Iterator* iter = db_->NewIterator(ReadOptions());
    int i = 0;
    int64_t bytes = 0;
    for (iter->SeekToFirst(); i < reads_ && iter->Valid(); iter->Next()) {
      bytes += iter->key().size() + iter->value().size();
      thread->stats.FinishedSingleOp();
      ++i;
    }
    delete iter;
    thread->stats.AddBytes(bytes);
  }

  void ReadReverse(ThreadState* thread) {
    Iterator* iter = db_->NewIterator(ReadOptions());
    int i = 0;
    int64_t bytes = 0;
    for (iter->SeekToLast(); i < reads_ && iter->Valid(); iter->Prev()) {
      bytes += iter->key().size() + iter->value().size();
      thread->stats.FinishedSingleOp();
      ++i;
    }
    delete iter;
    thread->stats.AddBytes(bytes);
  }

  void ReadRandom(ThreadState* thread) {
	uint64_t a, b, start, end;
    ReadOptions options;
    std::string value;
    int found = 0;
    micros(start);
    for (int i = 0; i < reads_; i++) {
      char key[100];
      const int k = thread->rand.Next() % FLAGS_num + FLAGS_base_key;
      snprintf(key, sizeof(key), "%016d", k);
      if (db_->Get(options, key, &value).ok()) {
        found++;
      }
      thread->stats.FinishedSingleOp();
    }
    char msg[100];
    snprintf(msg, sizeof(msg), "(%d of %d found)", found, num_);
    micros(end);
    print_timer_info("ReadRandom :: Total time taken to read all entries ", start, end);

    thread->stats.AddMessage(msg);
  }

  void ReadMissing(ThreadState* thread) {
    ReadOptions options;
    std::string value;
    for (int i = 0; i < reads_; i++) {
      char key[100];
      const int k = thread->rand.Next() % FLAGS_num;
      snprintf(key, sizeof(key), "%016d.", k);
      db_->Get(options, key, &value);
      thread->stats.FinishedSingleOp();
    }
  }

  void ReadHot(ThreadState* thread) {
    ReadOptions options;
    std::string value;
    const int range = (FLAGS_num + 99) / 100;
    for (int i = 0; i < reads_; i++) {
      char key[100];
      const int k = thread->rand.Next() % range;
      snprintf(key, sizeof(key), "%016d", k);
      db_->Get(options, key, &value);
      thread->stats.FinishedSingleOp();
    }
  }

  void SeekRandom(ThreadState* thread) {
	    printf("SeekRandom called. \n");
		uint64_t a, b, c, d, e;
	    ReadOptions options;
	    std::string value;
	    int found = 0;
	    micros(a);
	    std::map<int, int> micros_count;
	    for (int i = 0; i < reads_; i++) {
	      Iterator* iter = db_->NewIterator(options);
	      char key[100];
	      const int k = thread->rand.Next() % FLAGS_num;
	      snprintf(key, sizeof(key), "%016d", k);
	      iter->Seek(key);
	      if (iter->Valid() && iter->key() == key) {
	    	  found++;
	      } else {
	    	  if (iter->Valid()) {
	    		  printf("Key %s -- iter pointing to %s !\n", key, iter->key().data());
	    	  } else {
	    		  printf("Key %s -- iter not valid !\n", key);
	    	  }
	      }
	      delete iter;
	      thread->stats.FinishedSingleOp();
	    }
	    char msg[100];
	    snprintf(msg, sizeof(msg), "(%d of %d found)", found, reads_);
	    micros(b);
	    print_timer_info("SeekRandom:: Total time taken to seek N random values", a, b);
	    thread->stats.AddMessage(msg);
  }

  void ScanRandom(ThreadState* thread) {
	    printf("ScanRandom called. \n");
	    std::vector<int> next_sizes = {20, 40, 60, 80, 100};
	    int index = 0;
		uint64_t a, b, c, d, e;
		uint64_t seek_start, seek_end, seek_total = 0, scan_start, scan_end, scan_total = 0;
	    ReadOptions options;
	    std::string value;
	    int found = 0;
	    micros(a);
	    std::map<int, int> micros_count;
	    for (int i = 0; i < reads_; i++) {
	      Iterator* iter = db_->NewIterator(options);
	      char key[100];
	      const int k = thread->rand.Next() % FLAGS_num;
	      snprintf(key, sizeof(key), "%016d", k);
	      seek_start = Env::Default()->NowMicros();
	      iter->Seek(key);
	      seek_end = Env::Default()->NowMicros();
	      seek_total += seek_end - seek_start;
	      scan_start = Env::Default()->NowMicros();
	      int num_next = FLAGS_num_next;
	      if (iter->Valid()) {
	    	  if (iter->key() == key) {
	    		  found++;
	    	  }
	    	  for (int j = 0; j < num_next && iter->Valid(); j++) {
	    		  iter->Next();
	    	  }
	      }
	      scan_end = Env::Default()->NowMicros();
	      scan_total += scan_end - scan_start;
	      delete iter;
	      thread->stats.FinishedSingleOp();
	      index = (index + 1) % next_sizes.size();
	    }
	    char msg[100];
	    snprintf(msg, sizeof(msg), "(%d of %d found)", found, reads_);
	    micros(b);
	    printf("ScanRandom:: Time taken to seek N random values: %lu micros (%f ms)\n", seek_total, seek_total/1000.0);
	    printf("ScanRandom:: Time taken to scan num_next random values: %lu micros (%f ms)\n", scan_total, scan_total/1000.0);
	    print_timer_info("ScanRandom:: Total time taken to seek N random values", a, b);
	    thread->stats.AddMessage(msg);
  }

  void DoDelete(ThreadState* thread, bool seq) {
    RandomGenerator gen;
    WriteBatch batch;
    Status s;
    for (int i = 0; i < num_; i += entries_per_batch_) {
      batch.Clear();
      for (int j = 0; j < entries_per_batch_; j++) {
        const int k = seq ? i+j + FLAGS_base_key : (thread->rand.Next() % FLAGS_num) + FLAGS_base_key;
        char key[100];
        snprintf(key, sizeof(key), "%016d", k);
        batch.Delete(key);
        thread->stats.FinishedSingleOp();
      }
      s = db_->Write(write_options_, &batch);
      if (!s.ok()) {
        fprintf(stderr, "del error: %s\n", s.ToString().c_str());
        exit(1);
      }
    }
  }

  void DeleteSeq(ThreadState* thread) {
    DoDelete(thread, true);
  }

  void DeleteRandom(ThreadState* thread) {
    DoDelete(thread, false);
  }

  void SeekWhileWriting(ThreadState* thread) {
    if (thread->tid > 0) {
      SeekRandom(thread);
    } else {
      // Special thread that keeps writing until other threads are done.
      RandomGenerator gen;
      while (true) {
        {
          MutexLock l(&thread->shared->mu);
          if (thread->shared->num_done + 1 >= thread->shared->num_initialized) {
            // Other threads have finished
            break;
          }
        }

        const int k = thread->rand.Next() % FLAGS_num;
        char key[100];
        snprintf(key, sizeof(key), "%016d", k);
        Status s = db_->Put(write_options_, key, gen.Generate(value_size_));
        if (!s.ok()) {
          fprintf(stderr, "put error: %s\n", s.ToString().c_str());
          exit(1);
        }
      }

      // Do not count any of the preceding work/delay in stats.
      thread->stats.Start();
    }
  }

  void ReadWhileWriting(ThreadState* thread) {
    if (thread->tid > 0) {
      ReadRandom(thread);
    } else {
      // Special thread that keeps writing until other threads are done.
      uint64_t before_g, after_g;
      micros(before_g);
      RandomGenerator gen;
      int max_writes = 30000000;
      int num_writes = 0, num_actual_writes = 0;
      while (true) {
        {
          MutexLock l(&thread->shared->mu);
          if (thread->shared->num_done + 1 >= thread->shared->num_initialized) {
            // Other threads have finished
            break;
          }
        }

        const int k = thread->rand.Next() % FLAGS_num;
        char key[100];
        snprintf(key, sizeof(key), "%016d", k);
        if (num_writes < max_writes) {
			Status s = db_->Put(write_options_, key, gen.Generate(value_size_));
			if (!s.ok()) {
			  fprintf(stderr, "put error: %s\n", s.ToString().c_str());
			  exit(1);
			}
			num_actual_writes++;
        }
        num_writes++;
      }

      printf("Number of actual writes: %d\n", num_actual_writes);
      micros(after_g);
      print_timer_info("ReadwhileWriting() method :: Total time took to read all entries while writing other entries", after_g, before_g);
      // Do not count any of the preceding work/delay in stats.
      thread->stats.Start();
    }
  }

  void Compact(ThreadState* /*thread*/) {
    db_->CompactRange(NULL, NULL);
  }

  void PrintStats(const char* key) {
    std::string stats;
    if (!db_->GetProperty(key, &stats)) {
      stats = "(failed)";
    }
    fprintf(stdout, "\n%s\n", stats.c_str());
  }

  static void WriteToFile(void* arg, const char* buf, int n) {
    reinterpret_cast<WritableFile*>(arg)->Append(Slice(buf, n));
  }

  void HeapProfile() {
    char fname[100];
    snprintf(fname, sizeof(fname), "%s/heap-%04d", FLAGS_db, ++heap_counter_);
    WritableFile* file;
    Status s = Env::Default()->NewWritableFile(fname, &file);
    if (!s.ok()) {
      fprintf(stderr, "%s\n", s.ToString().c_str());
      return;
    }
    bool ok = port::GetHeapProfile(WriteToFile, file);
    delete file;
    if (!ok) {
      fprintf(stderr, "heap profiling not supported\n");
      Env::Default()->DeleteFile(fname);
    }
  }
};

}  // namespace leveldb

int main(int argc, char** argv) {
  FLAGS_write_buffer_size = leveldb::Options().write_buffer_size;
  FLAGS_open_files = leveldb::Options().max_open_files;
  FLAGS_block_size = leveldb::Options().block_size;
  std::string default_db_path;

  for (int i = 1; i < argc; i++) {
    double d;
    int n;
    char junk;
    if (leveldb::Slice(argv[i]).starts_with("--benchmarks=")) {
      FLAGS_benchmarks = argv[i] + strlen("--benchmarks=");
    } else if (sscanf(argv[i], "--compression_ratio=%lf%c", &d, &junk) == 1) {
      FLAGS_compression_ratio = d;
    } else if (sscanf(argv[i], "--histogram=%d%c", &n, &junk) == 1 &&
               (n == 0 || n == 1)) {
      FLAGS_histogram = n;
    } else if (sscanf(argv[i], "--use_existing_db=%d%c", &n, &junk) == 1 &&
               (n == 0 || n == 1)) {
      FLAGS_use_existing_db = n;
    } else if (sscanf(argv[i], "--num=%d%c", &n, &junk) == 1) {
      FLAGS_num = n;
    } else if (sscanf(argv[i], "--reads=%d%c", &n, &junk) == 1) {
      FLAGS_reads = n;
    } else if (sscanf(argv[i], "--threads=%d%c", &n, &junk) == 1) {
      FLAGS_threads = n;
    } else if (sscanf(argv[i], "--write_threads=%d%c", &n, &junk) == 1) {
      FLAGS_write_threads = n;
    } else if (sscanf(argv[i], "--read_threads=%d%c", &n, &junk) == 1) {
      FLAGS_read_threads = n;
    } else if (sscanf(argv[i], "--value_size=%d%c", &n, &junk) == 1) {
      FLAGS_value_size = n;
    } else if (sscanf(argv[i], "--write_buffer_size=%d%c", &n, &junk) == 1) {
      FLAGS_write_buffer_size = n;
    } else if (sscanf(argv[i], "--cache_size=%d%c", &n, &junk) == 1) {
      FLAGS_cache_size = n;
    } else if (sscanf(argv[i], "--block_size=%d%c", &n, &junk) == 1) {
      FLAGS_block_size = n;
    } else if (sscanf(argv[i], "--bloom_bits=%d%c", &n, &junk) == 1) {
      FLAGS_bloom_bits = n;
    } else if (sscanf(argv[i], "--open_files=%d%c", &n, &junk) == 1) {
      FLAGS_open_files = n;
    } else if (sscanf(argv[i], "--num_next=%d%c", &n, &junk) == 1) {
      FLAGS_num_next = n;
    } else if (sscanf(argv[i], "--base_key=%d%c", &n, &junk) == 1) {
      FLAGS_base_key = n;
    } else if (strncmp(argv[i], "--db=", 5) == 0) {
      FLAGS_db = argv[i] + 5;
    } else {
      fprintf(stderr, "Invalid flag '%s'\n", argv[i]);
      exit(1);
    }
  }

  // Choose a location for the test database if none given with --db=<path>
  if (FLAGS_db == NULL) {
      leveldb::Env::Default()->GetTestDirectory(&default_db_path);
      default_db_path += "/dbbench";
      FLAGS_db = default_db_path.c_str();
  }

  leveldb::Benchmark benchmark;
  benchmark.Run();
  return 0;
}
