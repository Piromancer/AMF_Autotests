#pragma once
#ifndef H_UTILITY_AUTOTESTS
#define H_UTILITY_AUTOTESTS
#include "../../include/core/Factory.h"
#include "../../common/AMFFactory.h"
#include "../../include/core/Buffer.h"
#include <gtest/gtest.h>
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <ctime>  
using namespace std;
using namespace amf;

#define CL_TARGET_OPENCL_VERSION 120

struct TestsInformation;

struct AllocationMetrics {
	uint32_t totalAllocated = 0;
	uint32_t totalFreed = 0;
	uint32_t totalPointersMade = 0;
	uint32_t totalPointersDestroyed = 0;

	uint32_t CurrentUsage();

	uint32_t CurrentPointers();
};

void* operator new(size_t size);

void operator delete(void* memory, size_t size);

bool has_suffix(const string& str, const string& suffix);

void initiateTestSuiteLog(string suiteName);

chrono::time_point<chrono::system_clock> initiateTestLog();

void terminateTestLog(chrono::time_point<chrono::system_clock> startTime);

void terminateTestSuiteLog();
#endif
