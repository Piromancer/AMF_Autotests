#include "autotests.h"

struct API : testing::Test {
	AMFFactoryHelper helper;
	AMFContextPtr context1;
	AMFComputeFactoryPtr oclComputeFactory;
	AMFFactory* factory;
	int deviceCount;
	chrono::time_point<chrono::system_clock> startTime;

	static void SetUpTestCase() {
		initiateTestSuiteLog("API");
	}

	static void TearDownTestCase() {
		terminateTestSuiteLog();
	}

	API() {
		helper.Init();
		factory = helper.GetFactory();
		factory->CreateContext(&context1);
		context1->SetProperty(AMF_CONTEXT_DEVICE_TYPE, AMF_CONTEXT_DEVICE_TYPE_GPU);
		context1->GetOpenCLComputeFactory(&oclComputeFactory);
		context1->InitOpenCL();
		deviceCount = oclComputeFactory->GetDeviceCount();
		g_AMFFactory.Init();
		startTime = initiateTestLog();
	}

	~API() {
		context1.Release();
		oclComputeFactory.Release();
		g_AMFFactory.Terminate();
		helper.Terminate();
		terminateTestLog(startTime);
	}
};

TEST_F(API, kernel_compute_complex) {
	g_AMFFactory.GetFactory()->SetCacheFolder(L"./cache");

	AMFPrograms* pPrograms;
	factory->GetPrograms(&pPrograms);

	AMF_KERNEL_ID kernel = 0;
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
		AMFComputeDevicePtr pComputeDevice;
		oclComputeFactory->GetDeviceAt(i, &pComputeDevice);
		pComputeDevice->GetNativeContext();

		AMFComputePtr pCompute;
		pComputeDevice->CreateCompute(nullptr, &pCompute);

		AMFComputeKernelPtr pKernel;
		res = pCompute->GetKernel(kernel, &pKernel);

		AMFBuffer* input = NULL;
		AMFBuffer* output = NULL;

		AMFContextPtr context;
		factory->CreateContext(&context);
		//context->InitOpenCLEx(pComputeDevice.GetPtr());
		context->InitOpenCL(pCompute->GetNativeCommandQueue());

		res = context->AllocBuffer(AMF_MEMORY_HOST, 1024 * sizeof(float), &input);
		res = context->AllocBuffer(AMF_MEMORY_OPENCL, 1024 * sizeof(float), &output);

		float* inputData = static_cast<float*>(input->GetNative());
		float* expectedData = new float[1024];
		for (int k = 0; k < 1024; k++)
		{
			inputData[k] = rand() / 50.00;
			expectedData[k] = inputData[k] * inputData[k];
		}

		input->Convert(AMF_MEMORY_OPENCL);

		res = pKernel->SetArgBuffer(1, output, AMF_ARGUMENT_ACCESS_WRITE);
		res = pKernel->SetArgBuffer(0, input, AMF_ARGUMENT_ACCESS_READ);
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
	}
}
TEST_F(API, kernel2_compute_complex_copenCl) {
	g_AMFFactory.GetFactory()->SetCacheFolder(L"./cache");

	AMFPrograms* pPrograms;
	factory->GetPrograms(&pPrograms);

	AMF_KERNEL_ID kernel = 0;
	const char* kernel_src = "\n" \
		"__kernel void square2( __global float* output, __global float* input, \n" \
		" const unsigned int count) {            \n" \
		" int i = get_global_id(0);              \n" \
		" if(i < count) \n" \
		" output[i] = input[i] * input[i]; \n" \
		"}                     \n" \
		"__kernel void multiplication(__global float* output, __global float* input, __global float* input2, \n" \
		" const unsigned int count) {            \n" \
		" int i = get_global_id(0);              \n" \
		" if(i < count) \n" \
		" output[i] = input[i] * input2[i]; \n" \
		"}                     \n";
	pPrograms->RegisterKernelSource(&kernel, L"kernelIDName", "multiplication", strlen(kernel_src), (amf_uint8*)kernel_src, NULL);

	for (int i = 0; i < deviceCount; ++i)
	{
		AMF_RESULT res;
		AMFComputeDevicePtr pComputeDevice;
		oclComputeFactory->GetDeviceAt(i, &pComputeDevice);
		pComputeDevice->GetNativeContext();

		AMFComputePtr pCompute;
		pComputeDevice->CreateCompute(nullptr, &pCompute);

		AMFComputeKernelPtr pKernel;
		res = pCompute->GetKernel(kernel, &pKernel);

		AMFBuffer* input = NULL;
		AMFBuffer* input2 = NULL;
		AMFBuffer* output = NULL;

		AMFContextPtr context;
		factory->CreateContext(&context);
		context->InitOpenCL(pCompute->GetNativeCommandQueue());

		res = context->AllocBuffer(AMF_MEMORY_HOST, 1024 * sizeof(float), &input);
		res = context->AllocBuffer(AMF_MEMORY_HOST, 1024 * sizeof(float), &input2);
		res = context->AllocBuffer(AMF_MEMORY_OPENCL, 1024 * sizeof(float), &output);

		float* inputData = static_cast<float*>(input->GetNative());
		float* inputData2 = static_cast<float*>(input2->GetNative());
		float* expectedData = new float[1024];
		for (int k = 0; k < 1024; k++)
		{
			inputData[k] = rand() / 50.00;
			inputData2[k] = rand() / 50.00;
			expectedData[k] = inputData[k] * inputData2[k];
		}

		input->Convert(AMF_MEMORY_OPENCL);

		res = pKernel->SetArgBuffer(0, output, AMF_ARGUMENT_ACCESS_WRITE);
		res = pKernel->SetArgBuffer(1, input, AMF_ARGUMENT_ACCESS_READ);
		res = pKernel->SetArgBuffer(2, input2, AMF_ARGUMENT_ACCESS_READ);
		res = pKernel->SetArgInt32(3, 1024);

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
	}
}
