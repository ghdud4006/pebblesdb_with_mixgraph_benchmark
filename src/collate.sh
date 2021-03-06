#!/bin/bash

echo ----------------------- Write ----------------------------
echo -n Total time taken to write all entries: 
cat $1 | grep 'WRITE_OVERALL_TIME' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo
echo -n Time taken in SequenceWriteBegin:' ' 
cat $1 | grep 'WRITE_SEQUENCE_WRITE_BEGIN_TOTAL' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo -n Time taken to set sequence and create new batch:' '
cat $1 | grep 'WRITE_SET_SEQUENCE_CREATE_NEW_BATCH' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo -n Time taken to set guards:' '
cat $1 | grep 'WRITE_SET_GUARDS' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo -n Time taken in Log.AddRecord:' '
cat $1 | grep 'LOG_ADDRECORD' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo -n Time taken to insert into version:' '
cat $1 | grep 'INSERT_INTO_VERSION' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo -n Time taken to sync log file:' '
cat $1 | grep 'WRITE_LOG_FILE_SYNC' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo -n Time taken in SequenceWriteEnd:' ' 
cat $1 | grep 'WRITE_SEQUENCE_WRITE_END_TOTAL' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo
echo '    '::SequenceWriteBegin::
echo -n Time taken to init mutex:' '
cat $1 | grep 'SWB_INIT_MUTEX' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo -n Time taken to init memtables:' '
cat $1 | grep 'SWB_INIT_MEMTABLES' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo -n Time taken to set log details:' '
cat $1 | grep 'SWB_SET_LOG_DETAILS' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo -n Time taken to enqueue mem:' '
cat $1 | grep 'SWB_ENQUEUE_MEM' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo -n Time taken to set tail:' '
cat $1 | grep 'SWB_SET_TAIL' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo -n Time taken to sync and fetch:' '
cat $1 | grep 'SWB_SYNC_AND_FETCH' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo
echo '    '::SequenceWriteEnd::
echo -n Time taken to lock mutex:' '
cat $1 | grep 'SWE_LOCK_MUTEX' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo -n Time taken to set next: ' '
cat $1 | grep 'SWE_SET_NEXT' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo -n Time taken to lock writers mutex: ' '
cat $1 | grep 'SWE_LOCK_WRITERS_MUTEX' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo -n Time taken to unlock writers mutex: ' '
cat $1 | grep 'SWE_UNLOCK_WRITERS_MUTEX' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo -n Time taken to set imm: ' '
cat $1 | grep 'SWE_SET_IMM' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo -n Time spent in sleep: ' '
cat $1 | grep 'SWE_SLEEP' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo 
echo ------------------------ Memtable compaction -------------------------
echo -n Total time spent in memtable compaction: ' '
cat $1 | grep 'TOTAL_MEMTABLE_COMPACTION' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo 
echo -n Time spent in WriteLevel0TableGuards: ' '
cat $1 | grep 'WRITE_LEVEL0_TABLE_GUARDS' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo -n Time spent in add guards to edit: ' '
cat $1 | grep 'CMT_ADD_GUARDS_TO_EDIT' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo -n Time spent in add complete guards to edit: ' '
cat $1 | grep 'CMT_ADD_COMPLETE_GUARDS_TO_EDIT' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo -n Time spent in LogAndApply: ' '
cat $1 | grep 'CMT_LOG_AND_APPLY' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo -n Time spent in erase pending outputs: ' '
cat $1 | grep 'CMT_ERASE_PENDING_OUTPUTS' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo -n Time spent in delete obsolete files: ' '
cat $1 | grep 'CMT_DELETE_OBSOLETE_FILES' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo
echo '    '::WriteLevel0TableGuards::
echo -n Time spent in BuildLevel0Tables: ' '
cat $1 | grep 'BUILD_LEVEL0_TABLES' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo -n Time spent in getting lock after building level0 tables: ' '
cat $1 | grep 'GET_LOCK_AFTER_BUILD_LEVEL0_TABLES' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo -n Time spent in adding level0 files to edit: ' '
cat $1 | grep 'ADD_LEVEL0_FILES_TO_EDIT' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo 
echo '    '::LogAndApply:: 
echo -n Apply edit: ' '
cat $1 | grep 'MTC_LAA_APPLY_EDIT' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo -n SaveTo: ' '
cat $1 | grep 'MTC_LAA_SAVETO' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo -n Finalize: ' '
cat $1 | grep 'MTC_LAA_FINALIZE' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo -n WriteSnapshot: ' '
cat $1 | grep 'MTC_LAA_COMPLETE_WRITE_SNAPSHOT' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo -n Encode edit: ' '
cat $1 | grep 'MTC_LAA_ENCODE_EDIT' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo -n Add record to manifest log: ' '
cat $1 | grep 'MTC_LAA_ADD_RECORD_TO' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo -n Sync manifest log file: ' '
cat $1 | grep 'MTC_LAA_SYNC_MANIFEST_' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo -n Get delta complete guards: ' '
cat $1 | grep 'MTC_LAA_GET_DELTA' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo -n Append version: ' '
cat $1 | grep 'MTC_LAA_APPEND' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo -n Sync complete guards: ' '
cat $1 | grep 'MTC_LAA_SYNC_COMPLETE_GUARDS' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo
echo '    '::SaveTo::
echo -n Add files: ' '
cat $1 | grep 'MTC_SAVETO_ADD_FILES' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo -n Add guards: ' '
cat $1 | grep 'MTC_SAVETO_ADD_GUARDS' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo -n Add complete guards: ' '
cat $1 | grep 'MTC_SAVETO_ADD_COMPLETE_GUARDS:' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo -n Add complete guards to guardset: ' '
cat $1 | grep 'MTC_SAVETO_ADD_COMPLETE_GUARDS_TO' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo -n Populate files: ' '
cat $1 | grep 'MTC_SAVETO_POPULATE_FILES' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo

echo -------------------------- Background compaction --------------------------------
echo -n Total time spent in Background compaction: ' '
cat $1 | grep 'TOTAL_BACKGROUND' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo
echo -n Pick compaction level: ' '
cat $1 | grep 'BGC_PICK_COMPACTION_LEVEL' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo -n Pick compaction: ' '
cat $1 | grep 'BGC_PICK_COMPACTION:' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo -n Add guards: ' '
cat $1 | grep 'BGC_ADD_GUARDS' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo -n Add complete_guards: ' '
cat $1 | grep 'BGC_ADD_COMPLETE' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo -n DoCompactionWorkGuards: ' '
cat $1 | grep 'BGC_DO_COMPACTION' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo -n Cleanup compaction: ' '
cat $1 | grep 'BGC_CLEANUP' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo
echo '    '::DoCompactionWorkGuards::
echo -n Make input iterator: ' '
cat $1 | grep 'BGC_MAKE_INPUT' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo -n Iterate through keys: ' '
cat $1 | grep 'BGC_ITERATE' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo -n Collect stats: ' '
cat $1 | grep 'BGC_COLLECT_STATS' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo -n Get lock before install: ' '
cat $1 | grep 'BGC_GET_LOCK' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo -n Install compaction results: ' '
cat $1 | grep 'BGC_INSTALL_COMPACTION' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo
echo '    '::InstallCompactionResult/LogAndApply::
echo -n Apply edit: ' '
cat $1 | grep 'BGC_LAA_APPLY_EDIT' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo -n SaveTo: ' '
cat $1 | grep 'BGC_LAA_SAVETO' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo -n Finalize: ' '
cat $1 | grep 'BGC_LAA_FINALIZE' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo -n WriteSnapshot: ' '
cat $1 | grep 'BGC_LAA_COMPLETE_WRITE_SNAPSHOT' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo -n Encode edit: ' '
cat $1 | grep 'BGC_LAA_ENCODE_EDIT' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo -n Add record to manifest log: ' '
cat $1 | grep 'BGC_LAA_ADD_RECORD_TO' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo -n Sync manifest log file: ' '
cat $1 | grep 'BGC_LAA_SYNC_MANIFEST_' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo -n Get delta complete guards: ' '
cat $1 | grep 'BGC_LAA_GET_DELTA' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo -n Append version: ' '
cat $1 | grep 'BGC_LAA_APPEND' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo -n Sync complete guards: ' '
cat $1 | grep 'BGC_LAA_SYNC_COMPLETE_GUARDS' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo
echo '    '::SaveTo::
echo -n Add files: ' '
cat $1 | grep 'BGC_SAVETO_ADD_FILES' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo -n Add guards: ' '
cat $1 | grep 'BGC_SAVETO_ADD_GUARDS' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo -n Add complete guards: ' '
cat $1 | grep 'BGC_SAVETO_ADD_COMPLETE_GUARDS:' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo -n Add complete guards to guardset: ' '
cat $1 | grep 'BGC_SAVETO_ADD_COMPLETE_GUARDS_TO' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo -n Populate files: ' '
cat $1 | grep 'BGC_SAVETO_POPULATE_FILES' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo

echo ------------------------------ Get -------------------------------------
echo -n Overall time taken to complete get: ' '
cat $1 | grep 'GET_OVERALL_TIME' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo
echo -n Time taken to get mutex: ' '
cat $1 | grep 'GET_TIME_TO_GET_MUTEX' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo -n Time taken to check mem: ' '
cat $1 | grep 'GET_TIME_TO_CHECK_MEM_IMM' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo -n Time taken to check version: ' '
cat $1 | grep 'GET_TIME_TO_CHECK_VERSION' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo -n Time taken to lock mutex: ' '
cat $1 | grep 'GET_TIME_TO_LOCK_MUTEX' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo -n Time to finish unref: ' '
cat $1 | grep 'GET_TIME_TO_FINISH_UNREF' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo 
echo '    '::GetFromVersion::
echo -n Find guard: ' '
cat $1 | grep 'GET_FIND_GUARD' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo -n Check sentinel files: ' '
cat $1 | grep 'GET_CHECK_SENTINEL' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo -n Sort sentinel files: ' '
cat $1 | grep 'GET_SORT_SENTINEL' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo -n Check guard files: ' '
cat $1 | grep 'GET_CHECK_GUAR' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo -n Sort guard files: ' '
cat $1 | grep 'GET_SORT_GUARD' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo -n Find list of files: ' '
cat $1 | grep 'GET_FIND_LIST_OF_FILES' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo -n Get from table cache: ' '
cat $1 | grep 'GET_TABLE_CACHE_GET:' | cut -d ' ' -f3 | awk '{s+=$1} END {print s}'
echo
echo -------------------------------------------------------------------------
