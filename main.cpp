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

TEST_F(Smoke, computeFactory_getDeviceCount) {
	EXPECT_EQ(oclComputeFactory->GetDeviceCount(), 1);
}

TEST_F(Smoke, computeFactory_getDeviceAt) {
	AMFComputeDevice* device;
	oclComputeFactory->GetDeviceAt(0, &device);
	EXPECT_TRUE(device);
}

TEST_F(Smoke, computeFactory_getDeviceAt_negative) {
	AMFComputeDevice* device;
	EXPECT_ANY_THROW(oclComputeFactory->GetDeviceAt(1000, &device));
}

TEST_F(Smoke, deviceCompute_getNativePlatform) {
	AMFComputeDevice* device;
	oclComputeFactory->GetDeviceAt(0, &device);
	EXPECT_TRUE(device->GetNativePlatform());
}

TEST_F(Smoke, deviceCompute_getNativeDeviceID) {
	AMFComputeDevice* device;
	oclComputeFactory->GetDeviceAt(0, &device);
	EXPECT_TRUE(device->GetNativeDeviceID());
}

TEST_F(Smoke, deviceCompute_getNativeContext) {
	AMFComputeDevice* device;
	oclComputeFactory->GetDeviceAt(0, &device);
	EXPECT_TRUE(device->GetNativeContext());
}

TEST_F(Smoke, deviceCompute_createCompute) {
	AMFComputeDevice* device;
	oclComputeFactory->GetDeviceAt(0, &device);
	AMFComputePtr pCompute;
	device->CreateCompute(nullptr, &pCompute);
	EXPECT_TRUE(pCompute);
}

TEST_F(Smoke, deviceCompute_createComputeEx) {
	AMFComputeDevice* device;
	oclComputeFactory->GetDeviceAt(0, &device);
	AMFComputePtr pCompute;
	device->CreateComputeEx(nullptr, &pCompute);
	EXPECT_TRUE(pCompute);
}

TEST_F(Smoke, programs_registerKernelSource) {
	AMFPrograms* program;
	factory->GetPrograms(&program);
	AMF_KERNEL_ID kernel = 0;
	const char* kernel_src = "\n" \
		"__kernel void square2( __global float* input, __global float* output, \n" \
		" const unsigned int count) {            \n" \
		" int i = get_global_id(0);              \n" \
		" if(i < count) \n" \
		" output[i] = input[i] * input[i]; \n" \
		"}                     \n";
	program->RegisterKernelSource(&kernel, L"kernelIDName", "square2", strlen(kernel_src), (amf_uint8*)kernel_src, NULL);
	AMFComputeKernelPtr pKernel;
	AMFComputeDevice* device;
	oclComputeFactory->GetDeviceAt(0, &device);
	AMFComputePtr pCompute;
	device->CreateCompute(nullptr, &pCompute);
	pCompute->GetKernel(kernel, &pKernel);
	EXPECT_TRUE(pKernel);
}
//TODO: Add kernel files
TEST_F(Smoke, DISABLED_programs_registerKernelSourceFile) {
	AMFPrograms* program;
	factory->GetPrograms(&program);
	AMF_KERNEL_ID kernel = 0;
	const char* kernel_src = "\n" \
		"__kernel void square2( __global float* input, __global float* output, \n" \
		" const unsigned int count) {            \n" \
		" int i = get_global_id(0);              \n" \
		" if(i < count) \n" \
		" output[i] = input[i] * input[i]; \n" \
		"}                     \n";
	program->RegisterKernelSource(&kernel, L"kernelIDName", "square2", strlen(kernel_src), (amf_uint8*)kernel_src, NULL);
	EXPECT_TRUE(kernel);
}

TEST_F(Smoke, DISABLED_programs_registerKernelBinary) {
	AMFPrograms* program;
	factory->GetPrograms(&program);
	AMF_KERNEL_ID kernel = 0;
	const char* kernel_src = "\n" \
		"__kernel void square2( __global float* input, __global float* output, \n" \
		" const unsigned int count) {            \n" \
		" int i = get_global_id(0);              \n" \
		" if(i < count) \n" \
		" output[i] = input[i] * input[i]; \n" \
		"}                     \n";
	program->RegisterKernelSource(&kernel, L"kernelIDName", "square2", strlen(kernel_src), (amf_uint8*)kernel_src, NULL);
	EXPECT_TRUE(kernel);
}

TEST_F(Smoke, compute_getMemoryType) {
	AMFComputeDevice* device;
	oclComputeFactory->GetDeviceAt(0, &device);
	AMFComputePtr pCompute;
	device->CreateCompute(nullptr, &pCompute);
	EXPECT_TRUE(pCompute->GetMemoryType());
}

TEST_F(Smoke, compute_getNativeContext) {
	AMFComputeDevice* device;
	oclComputeFactory->GetDeviceAt(0, &device);
	AMFComputePtr pCompute;
	device->CreateCompute(nullptr, &pCompute);
	EXPECT_TRUE(pCompute->GetNativeContext());
}

TEST_F(Smoke, compute_getNativeDeviceID) {
	AMFComputeDevice* device;
	oclComputeFactory->GetDeviceAt(0, &device);
	AMFComputePtr pCompute;
	device->CreateCompute(nullptr, &pCompute);
	EXPECT_TRUE(pCompute->GetNativeDeviceID());
}

TEST_F(Smoke, compute_getNativeCommandQueue) {
	AMFComputeDevice* device;
	oclComputeFactory->GetDeviceAt(0, &device);
	AMFComputePtr pCompute;
	device->CreateCompute(nullptr, &pCompute);
	EXPECT_TRUE(pCompute->GetNativeCommandQueue());
}

TEST_F(Smoke, compute_getKernel) {
	AMFComputeDevice* device;
	oclComputeFactory->GetDeviceAt(0, &device);
	AMFComputePtr pCompute;
	AMFPrograms* program;
	factory->GetPrograms(&program);
	AMF_KERNEL_ID kernel = 0;
	const char* kernel_src = "\n" \
		"__kernel void square2( __global float* input, __global float* output, \n" \
		" const unsigned int count) {            \n" \
		" int i = get_global_id(0);              \n" \
		" if(i < count) \n" \
		" output[i] = input[i] * input[i]; \n" \
		"}                     \n";
	program->RegisterKernelSource(&kernel, L"kernelIDName", "square2", strlen(kernel_src), (amf_uint8*)kernel_src, NULL);
	amf::AMFComputeKernelPtr pKernel;
	pCompute->GetKernel(kernel, &pKernel);
	EXPECT_TRUE(pKernel);
}

TEST_F(Smoke, compute_putSyncPoint) {
	AMFComputeDevice* device;
	oclComputeFactory->GetDeviceAt(0, &device);
	AMFComputePtr pCompute;
	device->CreateComputeEx(nullptr, &pCompute);
	AMFComputeSyncPointPtr sync;
	pCompute->PutSyncPoint(&sync);
	EXPECT_TRUE(sync);
}

TEST_F(Smoke, compute_flushQueue) {
	AMFComputeDevice* device;
	oclComputeFactory->GetDeviceAt(0, &device);
	AMFComputePtr pCompute;
	device->CreateComputeEx(nullptr, &pCompute);
	EXPECT_NO_THROW(pCompute->FlushQueue());
}

TEST_F(Smoke, compute_finishQueue) {
	AMFComputeDevice* device;
	oclComputeFactory->GetDeviceAt(0, &device);
	AMFComputePtr pCompute;
	device->CreateComputeEx(nullptr, &pCompute);
	EXPECT_NO_THROW(pCompute->FinishQueue());
}

TEST_F(Smoke, DISABLED_compute_fillPlane) {
	AMFComputeDevice* device;
	oclComputeFactory->GetDeviceAt(0, &device);
	AMFComputePtr pCompute;
	device->CreateComputeEx(nullptr, &pCompute);
	AMFSurfacePtr surface;
	context1->AllocSurface(AMF_MEMORY_OPENCL, AMF_SURFACE_RGBA, 2, 2, &surface);
	AMFPlanePtr plane = surface->GetPlane(AMF_PLANE_PACKED);
	amf_size origin[3] = { 0, 0, 0 };
	amf_size region[3] = { 1, 1, 0 };
	float color[4] = { 1, 1, 0, 0 };
	EXPECT_NO_THROW(pCompute->FillPlane(plane, origin, region, color));
}

TEST_F(Smoke, DISABLED_compute_fillBuffer) {
	AMFComputeDevice* device;
	oclComputeFactory->GetDeviceAt(0, &device);
	AMFComputePtr pCompute;
	device->CreateComputeEx(nullptr, &pCompute);
	EXPECT_TRUE(pCompute->GetMemoryType());
}

TEST_F(Smoke, compute_convertPlaneToBuffer) {
	AMFComputeDevice* device;
	oclComputeFactory->GetDeviceAt(0, &device);
	AMFComputePtr pCompute;
	device->CreateComputeEx(nullptr, &pCompute);
	AMFSurfacePtr surface;
	context1->AllocSurface(AMF_MEMORY_OPENCL, AMF_SURFACE_RGBA, 2, 2, &surface);
	AMFPlanePtr plane = surface->GetPlane(AMF_PLANE_PACKED);
	AMFBufferPtr buffer;
	pCompute->ConvertPlaneToBuffer(plane, &buffer);
	EXPECT_TRUE(buffer);
}

//TODO make those tests work properly
TEST_F(Smoke, DISABLED_compute_copyBuffer) {
	AMFComputeDevice* device;
	oclComputeFactory->GetDeviceAt(0, &device);
	AMFComputePtr pCompute;
	device->CreateComputeEx(nullptr, &pCompute);
	EXPECT_TRUE(pCompute->GetMemoryType());
}

TEST_F(Smoke, compute_copyPlane) {
	AMFComputeDevice* device;
	oclComputeFactory->GetDeviceAt(0, &device);
	AMFComputePtr pCompute;
	device->CreateComputeEx(nullptr, &pCompute);
	AMFSurfacePtr surface;
	context1->AllocSurface(AMF_MEMORY_OPENCL, AMF_SURFACE_RGBA, 2, 2, &surface);
	AMFPlanePtr plane = surface->GetPlane(AMF_PLANE_PACKED);
	AMFPlanePtr plane2;
	amf_size origin[3] = { 0, 0, 0 };
	amf_size region[3] = { 1, 1, 0 };
	float color[4] = { 1, 1, 0, 0 };
	pCompute->CopyPlane(plane, origin, region, plane2, origin);
	EXPECT_TRUE(plane2);
}

TEST_F(Smoke, DISABLED_compute_copyBufferToHost) {
	AMFComputeDevice* device;
	oclComputeFactory->GetDeviceAt(0, &device);
	AMFComputePtr pCompute;
	device->CreateComputeEx(nullptr, &pCompute);
	EXPECT_TRUE(pCompute->GetMemoryType());
}

TEST_F(Smoke, DISABLED_compute_copyBufferFromHost) {
	AMFComputeDevice* device;
	oclComputeFactory->GetDeviceAt(0, &device);
	AMFComputePtr pCompute;
	device->CreateComputeEx(nullptr, &pCompute);
	EXPECT_TRUE(pCompute->GetMemoryType());
}

TEST_F(Smoke, DISABLED_compute_copyPlaneToHost) {
	AMFComputeDevice* device;
	oclComputeFactory->GetDeviceAt(0, &device);
	AMFComputePtr pCompute;
	device->CreateComputeEx(nullptr, &pCompute);
	EXPECT_TRUE(pCompute->GetMemoryType());
}

TEST_F(Smoke, DISABLED_compute_copyPlaneFromHost) {
	AMFComputeDevice* device;
	oclComputeFactory->GetDeviceAt(0, &device);
	AMFComputePtr pCompute;
	device->CreateComputeEx(nullptr, &pCompute);
	EXPECT_TRUE(pCompute->GetMemoryType());
}

TEST_F(Smoke, DISABLED_compute_convertPlaneToPlane) {
	AMFComputeDevice* device;
	oclComputeFactory->GetDeviceAt(0, &device);
	AMFComputePtr pCompute;
	device->CreateComputeEx(nullptr, &pCompute);
	EXPECT_TRUE(pCompute->GetMemoryType());
}
