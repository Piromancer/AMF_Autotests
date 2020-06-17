#include "autotests.h"
#ifndef UTILITY_AUTOTESTS
#define UTILITY_AUTOTESTS

struct TestsInformation {
	uint32_t tests_ran = 0;
};

uint32_t AllocationMetrics::CurrentUsage() {
	return totalAllocated - totalFreed;
};

uint32_t AllocationMetrics::CurrentPointers() {
	return totalPointersMade - totalPointersDestroyed;
};

TestsInformation testsInfo;
AllocationMetrics memoryUsage; 
ofstream logFile;

void* operator new(size_t size) {
	memoryUsage.totalAllocated += size;
	memoryUsage.totalPointersMade++;

	return malloc(size);
}

void operator delete(void* memory, size_t size) {
	memoryUsage.totalFreed += size;
	memoryUsage.totalPointersDestroyed++;

	free(memory);
}

bool has_suffix(const string& str, const string& suffix)
{
	return str.size() >= suffix.size() &&
		str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

void initiateTestSuiteLog(string suiteName) {
	string spaces(29 - floor(suiteName.length()/2), ' ');
	logFile.open("out.log", ios::out | ios::app);
	logFile
		<< "|--------------------------------------------------------|" << endl
		<< spaces << suiteName << endl
		<< "|--------------------------------------------------------|" << endl;
	logFile.close();
}

chrono::time_point<chrono::system_clock> initiateTestLog() {
	logFile.open("out.log", ios::out | ios::app);
	chrono::time_point<chrono::system_clock> startTime = chrono::system_clock::now();
	time_t convertedTime = chrono::system_clock::to_time_t(startTime);
	logFile
		<< "Test case: " << ::testing::UnitTest::GetInstance()->current_test_info()->name() << endl
		<< "Time: " << std::ctime(&convertedTime) << endl
		<< "Memory metrics before test:" << endl
		<< "Memory usage - " << memoryUsage.CurrentUsage() << endl
		<< "Pointers count - " << memoryUsage.CurrentPointers() << endl;
	return startTime;
}

void terminateTestLog(chrono::time_point<chrono::system_clock> startTime) {
	auto endTime = chrono::system_clock::now();
	chrono::duration<double> elapsed_seconds = endTime - startTime;
	logFile
		<< "Time elapsed: " << elapsed_seconds.count() << " s" << endl
		<< "Memory metrics after test:" << endl
		<< "Memory usage - " << memoryUsage.CurrentUsage() << endl
		<< "Pointers count - " << memoryUsage.CurrentPointers() << endl
		<< "----------------------------------------------------------" << endl;
	logFile.close();
}

void terminateTestSuiteLog() {
	logFile.open("out.log", ios::out | ios::app);
	logFile
		<< endl
		<< endl;
	logFile.close();
}
#endif