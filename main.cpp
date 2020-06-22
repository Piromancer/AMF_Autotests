#include "autotests.h"

struct Smoke : testing::Test {
	AMFFactoryHelper helper;
	AMFContextPtr context1;
	AMFComputeFactoryPtr oclComputeFactory;
	AMFFactory* factory;
	int deviceCount;
	chrono::time_point<chrono::system_clock> startTime;

	static void SetUpTestCase() {
		initiateTestSuiteLog("Smoke");
	}

	static void TearDownTestCase() {
		terminateTestSuiteLog();
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
		startTime = initiateTestLog();
	}

	~Smoke() {
		context1.Release();
		oclComputeFactory.Release();
		g_AMFFactory.Terminate();
		helper.Terminate();
		terminateTestLog(startTime);
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

TEST_F(Smoke, rect_test) {
	AMFRect rect{ 0, 1, 1, 0 };
	EXPECT_EQ(rect.Height(), -rect.Width());
	EXPECT_EQ(rect.Height(), -1);
}

TEST_F(Smoke, get_compute) {
	AMFCompute* compute;
	context1->GetCompute(AMF_MEMORY_OPENCL, &compute);
	EXPECT_TRUE(compute);
}

TEST_F(Smoke, get_device_at) {
	AMFComputeDevice* device;
	oclComputeFactory->GetDeviceAt(0, &device);
	amf::AMFComputePtr pCompute;
	device->CreateCompute(nullptr, &pCompute);
	EXPECT_TRUE(device);
	EXPECT_TRUE(pCompute);
}

TEST_F(Smoke, get_device_at_negative) {
	AMFComputeDevice* device;
	EXPECT_ANY_THROW(oclComputeFactory->GetDeviceAt(1000, &device));
}
