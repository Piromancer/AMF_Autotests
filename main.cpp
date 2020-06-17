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

struct TestsInformation {
	uint32_t tests_ran = 0;
};

struct AllocationMetrics {
	uint32_t totalAllocated = 0;
	uint32_t totalFreed = 0;
	uint32_t totalPointersMade = 0;
	uint32_t totalPointersDestroyed = 0;

	uint32_t CurrentUsage() {
		return totalAllocated - totalFreed;
	}

	uint32_t CurrentPointers() {
		return totalPointersMade - totalPointersDestroyed;
	}
};

static TestsInformation testsInfo;
static AllocationMetrics memoryUsage;

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

struct Smoke : testing::Test {
	AMFFactoryHelper helper;
	AMFContextPtr context1;
	AMFComputeFactoryPtr oclComputeFactory;
	AMFFactory* factory;
	ofstream logFile;
	int deviceCount;

	static void SetUpTestCase() {
		ofstream logFile;
		logFile.open("out.log", ios::out | ios::app);
		logFile
			<< "|--------------------------------------------------------|" << endl
			<< "                     Smoke group started" << endl
			<< "|--------------------------------------------------------|" << endl;
		logFile.close();
	}

	static void TearDownTestCase() {
		ofstream logFile;
		logFile.open("out.log", ios::out | ios::app);
		logFile 
			<< endl 
			<< endl;
		logFile.close();
	}

	Smoke() {
		helper.Init();
		factory = helper.GetFactory();
		factory->CreateContext(&context1);
		context1->SetProperty(AMF_CONTEXT_DEVICE_TYPE, AMF_CONTEXT_DEVICE_TYPE_GPU);
		context1->GetOpenCLComputeFactory(&oclComputeFactory);
		context1->InitOpenCL();
		deviceCount = oclComputeFactory->GetDeviceCount();
		g_AMFFactory.Init();
		g_AMFFactory.GetDebug()->AssertsEnable(true);
		g_AMFFactory.GetTrace()->SetWriterLevel(AMF_TRACE_WRITER_FILE, AMF_TRACE_TRACE);
		g_AMFFactory.GetTrace()->SetGlobalLevel(AMF_TRACE_TRACE);
		g_AMFFactory.GetTrace()->SetWriterLevel(AMF_TRACE_WRITER_CONSOLE, AMF_TRACE_TRACE);
		g_AMFFactory.GetTrace()->SetWriterLevelForScope(AMF_TRACE_WRITER_CONSOLE, L"scope2", AMF_TRACE_TRACE);
		g_AMFFactory.GetTrace()->SetWriterLevelForScope(AMF_TRACE_WRITER_CONSOLE, L"scope2", AMF_TRACE_ERROR);
		logFile.open("out.log", ios::out | ios::app);
		auto startTime = chrono::system_clock::now();
		time_t convertedTime = chrono::system_clock::to_time_t(startTime);
		logFile 
			<< "Time: " << std::ctime(&convertedTime) << endl
			<< "Test case: "<<::testing::UnitTest::GetInstance()->current_test_info()->name() << endl
			<< "Before test:" << endl
			<< "Memory usage - " << memoryUsage.CurrentUsage() << endl
			<< "Pointers count - " << memoryUsage.CurrentPointers() << endl;
	}

	~Smoke() {
		context1.Release();
		oclComputeFactory.Release();
		g_AMFFactory.Terminate();
		helper.Terminate();
		logFile 
			<< "After test:" << endl
			<< "Memory usage - " << memoryUsage.CurrentUsage() << endl
			<< "Pointers count - " << memoryUsage.CurrentPointers() << endl
			<< "----------------------------------------------------------" << endl;
		logFile.close();
	}
};

TEST_F(Smoke, set_cache_folder) {
	g_AMFFactory.GetFactory()->SetCacheFolder(L"cache");
	EXPECT_STREQ(g_AMFFactory.GetFactory()->GetCacheFolder(), L"cache");
}

TEST_F(Smoke, release_null_check) {
	context1.Release();
	EXPECT_EQ(context1, (amf::AMFContextPtr)NULL);
}

TEST_F(Smoke, traceW_error) {
	g_AMFFactory.GetTrace()->SetPath(L"traceW.log");
	g_AMFFactory.GetTrace()->TraceW(L"path", 387, AMF_TRACE_ERROR, L"scope", 4, L"Error message");
	fstream fd;
	fd.open("traceW.log", ios::in);
	string log;
	getline(fd, log);
	EXPECT_TRUE(has_suffix(log, (string)"Error message"));
}

TEST_F(Smoke, kernel_compute_complex) {

	amf::AMFPrograms* pPrograms;
	factory->GetPrograms(&pPrograms);

	amf::AMF_KERNEL_ID kernel = 0;
	const char* kernel_src = "\n" \
		"__kernel void square2( __global float* input, __global float* output, \n" \
		" const unsigned int count) {            \n" \
		" int i = get_global_id(0);              \n" \
		" if(i < count) \n" \
		" output[i] = input[i] * input[i]; \n" \
		"}                     \n";
	pPrograms->RegisterKernelSource(&kernel, L"kernelIDName", "square2", strlen(kernel_src), (amf_uint8*)kernel_src, NULL);

	for (int i = 0; i < deviceCount; ++i)
	{
		AMF_RESULT res;
		amf::AMFComputeDevicePtr pComputeDevice;
		oclComputeFactory->GetDeviceAt(i, &pComputeDevice);
		pComputeDevice->GetNativeContext();

		amf::AMFComputePtr pCompute;
		pComputeDevice->CreateCompute(nullptr, &pCompute);

		amf::AMFComputeKernelPtr pKernel;
		res = pCompute->GetKernel(kernel, &pKernel);

		amf::AMFBuffer* input = NULL;
		amf::AMFBuffer* output = NULL;

		amf::AMFContextPtr context;
		factory->CreateContext(&context);
		//context->InitOpenCLEx(pComputeDevice.GetPtr());
		context->InitOpenCL(pCompute->GetNativeCommandQueue());

		res = context->AllocBuffer(amf::AMF_MEMORY_HOST, 1024 * sizeof(float), &input);
		res = context->AllocBuffer(amf::AMF_MEMORY_OPENCL, 1024 * sizeof(float), &output);

		float* inputData = static_cast<float*>(input->GetNative());
		float* expectedData = new float[1024];
		for (int k = 0; k < 1024; k++)
		{
			inputData[k] = rand() / 50.00;
			expectedData[k] = inputData[k] * inputData[k];
		}

		input->Convert(amf::AMF_MEMORY_OPENCL);

		res = pKernel->SetArgBuffer(1, output, amf::AMF_ARGUMENT_ACCESS_WRITE);
		res = pKernel->SetArgBuffer(0, input, amf::AMF_ARGUMENT_ACCESS_READ);
		res = pKernel->SetArgInt32(2, 1024);

		amf_size sizeLocal[3] = { 1024, 0, 0 };
		amf_size sizeGlobal[3] = { 1024, 0, 0 };
		amf_size offset[3] = { 0, 0, 0 };

		pKernel->GetCompileWorkgroupSize(sizeLocal);

		pKernel->Enqueue(1, offset, sizeGlobal, sizeLocal);
		pCompute->FlushQueue();
		pCompute->FinishQueue();
		float* outputData2 = NULL;
		res = output->MapToHost((void**)&outputData2, 0, 1024 * sizeof(float), true);


		for (int k = 0; k < 1024; k++)
		{
			EXPECT_LE(abs(expectedData[k] - outputData2[k]), 0.01);
		}

		output->Convert(amf::AMF_MEMORY_HOST);
		float* outputData = static_cast<float*>(output->GetNative());
	}
}
