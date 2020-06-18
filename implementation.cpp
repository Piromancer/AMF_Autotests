#include "autotests.h"

struct Implementation : testing::Test {
	AMFFactoryHelper helper;
	AMFContextPtr context1;
	AMFComputeFactoryPtr oclComputeFactory;
	AMFFactory* factory;
	int deviceCount;
	chrono::time_point<chrono::system_clock> startTime;

	static void SetUpTestCase() {
		initiateTestSuiteLog("Implementation");
	}

	static void TearDownTestCase() {
		terminateTestSuiteLog();
	}

	Implementation() {
		helper.Init();
		factory = helper.GetFactory();
		factory->CreateContext(&context1);
		g_AMFFactory.Init();
		startTime = initiateTestLog();
	}

	~Implementation() {
		context1.Release();
		oclComputeFactory.Release();
		g_AMFFactory.Terminate();
		helper.Terminate();
		terminateTestLog(startTime);
	}
};

TEST_F(Implementation, openCL_compute_factory_impl) {
	AMFComputeFactoryPtr fact;
	AMF_RESULT res = context1->GetOpenCLComputeFactory(&fact);
	EXPECT_NE(res, AMF_NOT_IMPLEMENTED);
}

TEST_F(Implementation, DISABLED_DX11_impl) {
	AMF_RESULT res = context1->InitDX11(context1->GetDX11Device());
	EXPECT_NE(res, AMF_NOT_IMPLEMENTED);
}

TEST_F(Implementation, DISABLED_DX9_impl) {
	AMF_RESULT res = context1->InitDX9(context1->GetDX9Device());
	EXPECT_NE(res, AMF_NOT_IMPLEMENTED);
}

TEST_F(Implementation, DISABLED_openGL_impl) {
	AMF_RESULT res = context1->InitOpenGL(context1->GetOpenGLContext(), NULL, NULL);
	EXPECT_NE(res, AMF_NOT_IMPLEMENTED);
}

TEST_F(Implementation, DISABLED_xv_impl) {
	AMF_RESULT res = context1->InitXV(context1->GetXVDevice());
	EXPECT_NE(res, AMF_NOT_IMPLEMENTED);
}

TEST_F(Implementation, DISABLED_grall_impl) {
	AMF_RESULT res = context1->InitGralloc(context1->GetGrallocDevice());
	EXPECT_NE(res, AMF_NOT_IMPLEMENTED);
}

TEST_F(Implementation, DISABLED_opencl_lock_unlock_impl) {
	AMF_RESULT res = context1->LockOpenCL();
	EXPECT_NE(res, AMF_NOT_IMPLEMENTED);
	res = context1->UnlockOpenCL();
	EXPECT_NE(res, AMF_NOT_IMPLEMENTED);
}

