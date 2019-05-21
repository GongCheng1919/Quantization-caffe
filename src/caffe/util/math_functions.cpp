#include <boost/math/special_functions/next.hpp>
#include <boost/random.hpp>

#include <limits>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template<>
void caffe_cpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
}

template<>
void caffe_cpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_dgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
}

template <>
void caffe_cpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
  cblas_sgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

template <>
void caffe_cpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, const double* x,
    const double beta, double* y) {
  cblas_dgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

template <>
void caffe_axpy<float>(const int N, const float alpha, const float* X,
    float* Y) { cblas_saxpy(N, alpha, X, 1, Y, 1); }

template <>
void caffe_axpy<double>(const int N, const double alpha, const double* X,
    double* Y) { cblas_daxpy(N, alpha, X, 1, Y, 1); }

template <typename Dtype>
void caffe_set(const int N, const Dtype alpha, Dtype* Y) {
  if (alpha == 0) {
    memset(Y, 0, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
    return;
  }
  for (int i = 0; i < N; ++i) {
    Y[i] = alpha;
  }
}

template void caffe_set<int>(const int N, const int alpha, int* Y);
template void caffe_set<float>(const int N, const float alpha, float* Y);
template void caffe_set<double>(const int N, const double alpha, double* Y);

template <>
void caffe_add_scalar(const int N, const float alpha, float* Y) {
  for (int i = 0; i < N; ++i) {
    Y[i] += alpha;
  }
}

template <>
void caffe_add_scalar(const int N, const double alpha, double* Y) {
  for (int i = 0; i < N; ++i) {
    Y[i] += alpha;
  }
}

template <typename Dtype>
void caffe_copy(const int N, const Dtype* X, Dtype* Y) {
  if (X != Y) {
    if (Caffe::mode() == Caffe::GPU) {
#ifndef CPU_ONLY
      // NOLINT_NEXT_LINE(caffe/alt_fn)
      CUDA_CHECK(cudaMemcpy(Y, X, sizeof(Dtype) * N, cudaMemcpyDefault));
#else
      NO_GPU;
#endif
    } else {
      memcpy(Y, X, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
    }
  }
}

template void caffe_copy<int>(const int N, const int* X, int* Y);
template void caffe_copy<unsigned int>(const int N, const unsigned int* X,
    unsigned int* Y);
template void caffe_copy<float>(const int N, const float* X, float* Y);
template void caffe_copy<double>(const int N, const double* X, double* Y);

template <>
void caffe_scal<float>(const int N, const float alpha, float *X) {
  cblas_sscal(N, alpha, X, 1);
}

template <>
void caffe_scal<double>(const int N, const double alpha, double *X) {
  cblas_dscal(N, alpha, X, 1);
}

template <>
void caffe_cpu_axpby<float>(const int N, const float alpha, const float* X,
                            const float beta, float* Y) {
  cblas_saxpby(N, alpha, X, 1, beta, Y, 1);
}

template <>
void caffe_cpu_axpby<double>(const int N, const double alpha, const double* X,
                             const double beta, double* Y) {
  cblas_daxpby(N, alpha, X, 1, beta, Y, 1);
}

template <>
void caffe_add<float>(const int n, const float* a, const float* b,
    float* y) {
  vsAdd(n, a, b, y);
}

template <>
void caffe_add<double>(const int n, const double* a, const double* b,
    double* y) {
  vdAdd(n, a, b, y);
}

template <>
void caffe_sub<float>(const int n, const float* a, const float* b,
    float* y) {
  vsSub(n, a, b, y);
}

template <>
void caffe_sub<double>(const int n, const double* a, const double* b,
    double* y) {
  vdSub(n, a, b, y);
}

template <>
void caffe_mul<float>(const int n, const float* a, const float* b,
    float* y) {
  vsMul(n, a, b, y);
}

template <>
void caffe_mul<double>(const int n, const double* a, const double* b,
    double* y) {
  vdMul(n, a, b, y);
}

template <>
void caffe_div<float>(const int n, const float* a, const float* b,
    float* y) {
  vsDiv(n, a, b, y);
}

template <>
void caffe_div<double>(const int n, const double* a, const double* b,
    double* y) {
  vdDiv(n, a, b, y);
}

template <>
void caffe_powx<float>(const int n, const float* a, const float b,
    float* y) {
  vsPowx(n, a, b, y);
}

template <>
void caffe_powx<double>(const int n, const double* a, const double b,
    double* y) {
  vdPowx(n, a, b, y);
}

template <>
void caffe_sqr<float>(const int n, const float* a, float* y) {
  vsSqr(n, a, y);
}

template <>
void caffe_sqr<double>(const int n, const double* a, double* y) {
  vdSqr(n, a, y);
}

template <>
void caffe_sqrt<float>(const int n, const float* a, float* y) {
  vsSqrt(n, a, y);
}

template <>
void caffe_sqrt<double>(const int n, const double* a, double* y) {
  vdSqrt(n, a, y);
}

template <>
void caffe_exp<float>(const int n, const float* a, float* y) {
  vsExp(n, a, y);
}

template <>
void caffe_exp<double>(const int n, const double* a, double* y) {
  vdExp(n, a, y);
}

template <>
void caffe_log<float>(const int n, const float* a, float* y) {
  vsLn(n, a, y);
}

template <>
void caffe_log<double>(const int n, const double* a, double* y) {
  vdLn(n, a, y);
}

template <>
void caffe_abs<float>(const int n, const float* a, float* y) {
    vsAbs(n, a, y);
}

template <>
void caffe_abs<double>(const int n, const double* a, double* y) {
    vdAbs(n, a, y);
}

unsigned int caffe_rng_rand() {
  return (*caffe_rng())();
}

template <typename Dtype>
Dtype caffe_nextafter(const Dtype b) {
  return boost::math::nextafter<Dtype>(
      b, std::numeric_limits<Dtype>::max());
}

template
float caffe_nextafter(const float b);

template
double caffe_nextafter(const double b);

template <typename Dtype>
void caffe_rng_uniform(const int n, const Dtype a, const Dtype b, Dtype* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_LE(a, b);
  boost::uniform_real<Dtype> random_distribution(a, caffe_nextafter<Dtype>(b));
  boost::variate_generator<caffe::rng_t*, boost::uniform_real<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template
void caffe_rng_uniform<float>(const int n, const float a, const float b,
                              float* r);

template
void caffe_rng_uniform<double>(const int n, const double a, const double b,
                               double* r);

template <typename Dtype>
void caffe_rng_gaussian(const int n, const Dtype a,
                        const Dtype sigma, Dtype* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GT(sigma, 0);
  boost::normal_distribution<Dtype> random_distribution(a, sigma);
  boost::variate_generator<caffe::rng_t*, boost::normal_distribution<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template
void caffe_rng_gaussian<float>(const int n, const float mu,
                               const float sigma, float* r);

template
void caffe_rng_gaussian<double>(const int n, const double mu,
                                const double sigma, double* r);

template <typename Dtype>
void caffe_rng_bernoulli(const int n, const Dtype p, int* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GE(p, 0);
  CHECK_LE(p, 1);
  boost::bernoulli_distribution<Dtype> random_distribution(p);
  boost::variate_generator<caffe::rng_t*, boost::bernoulli_distribution<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template
void caffe_rng_bernoulli<double>(const int n, const double p, int* r);

template
void caffe_rng_bernoulli<float>(const int n, const float p, int* r);

template <typename Dtype>
void caffe_rng_bernoulli(const int n, const Dtype p, unsigned int* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GE(p, 0);
  CHECK_LE(p, 1);
  boost::bernoulli_distribution<Dtype> random_distribution(p);
  boost::variate_generator<caffe::rng_t*, boost::bernoulli_distribution<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = static_cast<unsigned int>(variate_generator());
  }
}

template
void caffe_rng_bernoulli<double>(const int n, const double p, unsigned int* r);

template
void caffe_rng_bernoulli<float>(const int n, const float p, unsigned int* r);

template <>
float caffe_cpu_strided_dot<float>(const int n, const float* x, const int incx,
    const float* y, const int incy) {
  return cblas_sdot(n, x, incx, y, incy);
}

template <>
double caffe_cpu_strided_dot<double>(const int n, const double* x,
    const int incx, const double* y, const int incy) {
  return cblas_ddot(n, x, incx, y, incy);
}

template <typename Dtype>
Dtype caffe_cpu_dot(const int n, const Dtype* x, const Dtype* y) {
  return caffe_cpu_strided_dot(n, x, 1, y, 1);
}

template
float caffe_cpu_dot<float>(const int n, const float* x, const float* y);

template
double caffe_cpu_dot<double>(const int n, const double* x, const double* y);

template <>
float caffe_cpu_asum<float>(const int n, const float* x) {
  return cblas_sasum(n, x, 1);
}

template <>
double caffe_cpu_asum<double>(const int n, const double* x) {
  return cblas_dasum(n, x, 1);
}

template <>
void caffe_cpu_scale<int>(const int n, const int alpha, const int *x,
                            int* y) {
}

template <>
void caffe_cpu_scale<unsigned>(const int n, const unsigned alpha, const unsigned *x,
                             unsigned* y) {
}

template <>
void caffe_cpu_scale<float>(const int n, const float alpha, const float *x,
                            float* y) {
  cblas_scopy(n, x, 1, y, 1);
  cblas_sscal(n, alpha, y, 1);
}

template <>
void caffe_cpu_scale<double>(const int n, const double alpha, const double *x,
                             double* y) {
  cblas_dcopy(n, x, 1, y, 1);
  cblas_dscal(n, alpha, y, 1);
}

template<>
void caffe_cpu_ternary<int>(const int N, const int delta, const int* X, int* Y,int* alpha){}

template<>
void caffe_cpu_ternary<unsigned>(const int N, const unsigned delta, const unsigned* X, unsigned* Y,unsigned* alpha){}

template<>
void caffe_cpu_ternary<float>(const int N, const float delta, const float* X, float* Y,float* alpha){
	float alpha_tmp=0;
	int num=0;
	for(int i=0; i<N; i++){
		float x = X[i];
		Y[i] = (x>delta) - (x<-delta);
		alpha_tmp+=Y[i]*x;
		num+=Y[i]*Y[i];
	}
	if(num==0){
		*alpha=0;
		return;
	}
	*alpha=alpha_tmp/num;
}

template<>
void caffe_cpu_ternary<double>(const int N, const double delta, const double* X, double* Y,double* alpha){
	double alpha_tmp=0;
	int num=0;
	for(int i=0; i<N; i++){
		double x = X[i];
		Y[i] = (x>delta) - (x<-delta);
		alpha_tmp+=Y[i]*x;
		num+=Y[i]*Y[i];
	}
	if(num==0){
		*alpha=0;
		return;
	}
	*alpha=alpha_tmp/num;
}
template<>
void caffe_cpu_quantizea<int>(const int count, const int* X, int* Y, int* fixed_point, int max_bits, bool calc_fixed_point){}
template<>
void caffe_cpu_quantizea<unsigned>(const int count, const unsigned* X, unsigned* Y, int* fixed_point, int max_bits, bool calc_fixed_point){}
template<>
void caffe_cpu_quantizea<float>(const int count, const float* X, float* Y, int* fixed_point, int max_bits, bool calc_fixed_point){
	if(calc_fixed_point){
		float max_n = fabsf(X[0]);
		int mindex=0;
		//用cblas求最大绝对值
		mindex=cblas_isamax(count,X,1);
		//LOG(INFO)<<fabsf(X[mindex+1])<<","<<fabsf(X[mindex])<<","<<fabsf(X[mindex-1]);
		max_n=fabsf(X[mindex]);
		/*
		for (int i = 1; i < count; i++){
			const float absx = fabsf(X[i]);
			//if (absx < min_n){ min_n = absx; }
			if (absx >max_n){ max_n = absx; }
		}*/
		if (max_n == 0){
			*fixed_point = 0;
		}else{
			*fixed_point = floor(log2(max_n)) - (max_bits - 2);//-2考虑符号位
		}
	}
	//std::cout<<",X[0]="<<X[0]<<",fabsf X[0]="<<fabsf(X[0])<<",max_n="<<max_n<<std::endl;
	const float max_num = pow(2, max_bits-1);
	const float min_num = -max_num;
	const float fixed_value = pow(2, *fixed_point);
	//生成随机数，用于随机化量化
	//boost::uniform_real<float> random_distribution(-0.5, caffe_nextafter<float>(0.5));
	//boost::variate_generator<caffe::rng_t*, boost::uniform_real<float> >
      //variate_generator(caffe_rng(), random_distribution);
	//for (int i = 0; i < n; ++i) {
    //r[i] = variate_generator();
	//}
	
	for (int i = 0; i < count; i++){
		float x_q = X[i] / fixed_value;
		//float randN=variate_generator();
		//x_q+=randN;
		x_q = (x_q > 0.0) ? floor(x_q + 0.5) : ceil(x_q - 0.5);
		if (x_q >= max_num)
		{
			Y[i]= max_num - 1;
		}else if(x_q <= min_num){
			Y[i]= min_num + 1;
		}else {	Y[i] = x_q;}
	}
}
template<>
void caffe_cpu_quantizea<double>(const int count, const double* X, double* Y, int* fixed_point, int max_bits, bool calc_fixed_point){
	if(calc_fixed_point){
		double max_n = fabs(X[0]);
		//用cblas求最大绝对值
		int mindex=cblas_idamax(count,X,1);
		//LOG(INFO)<<fabs(X[mindex])<<","<<fabs(X[mindex-1]);
		max_n=fabs(X[mindex]);
		/*
		for (int i = 1; i < count; i++){
			const double absx = fabs(X[i]);
			//if (absx < min_n){ min_n = absx; }
			if (absx >max_n){ max_n = absx; }
		}*/
		if (max_n == 0){
			*fixed_point = 0;
		}else{
			*fixed_point = floor(log2(max_n)) - (max_bits - 2);
		}
	}
	const double max_num = pow(2, max_bits-1);
	const double min_num = -max_num;
	const double fixed_value = pow(2, *fixed_point);
	//生成随机数，用于随机化量化
	//boost::uniform_real<double> random_distribution(-0.5, caffe_nextafter<double>(0.5));
	//boost::variate_generator<caffe::rng_t*, boost::uniform_real<double> >
    //  variate_generator(caffe_rng(), random_distribution);
	for (int i = 0; i < count; i++){
		double x_q = X[i] / fixed_value;
	//	double randN=variate_generator();
	//	x_q+=randN;
		//round
		x_q = (x_q > 0.0) ? floor(x_q + 0.5) : ceil(x_q - 0.5);
		if (x_q >= max_num)
		{
			Y[i]= max_num - 1;
		}else if(x_q <= min_num){
			Y[i]= min_num + 1;
		}else {	Y[i] = x_q;}
	}
}


//quantization——planb
template<>
void caffe_cpu_quantizeb<int>(const int count, int* X, int max_bits){}
template<>
void caffe_cpu_quantizeb<unsigned>(const int count, unsigned* X, int max_bits){}
template<>
void caffe_cpu_quantizeb<float>(const int count, float* X, int max_bits){
	//实现剪切
	float max_pos=(float)pow(2,max_bits-1)-1;
	float max_neg=-(float)pow(2,max_bits-1);
	for(int i=0;i<count;i++){
		//std::cout<<"value="<<X[i]<<",max_pos="<<max_pos<<",max_neg="<<max_neg<<std::endl;
		if(X[i]>max_pos){
			X[i]=max_pos;
		}
		if(X[i]<max_neg){
			X[i]=max_neg;
		}
	}
}
template<>
void caffe_cpu_quantizeb<double>(const int count, double* X, int max_bits){
	//实现剪切
	double max_pos=(double)pow(2,max_bits-1)-1;
	double max_neg=-(double)pow(2,max_bits-1);
	for(int i=0;i<count;i++){
		//std::cout<<"value="<<X[i]<<",max_pos="<<max_pos<<",max_neg="<<max_neg<<std::endl;
		if(X[i]>max_pos){
			X[i]=max_pos;
		}
		if(X[i]<max_neg){
			X[i]=max_neg;
		}
	}
}
//scale clip
template<>
void caffe_cpu_quantizec<int>(const int count, const int* X, int* ,int* alpha, int max_bits){}
template<>
void caffe_cpu_quantizec<unsigned>(const int count,const unsigned* X, unsigned* Y, unsigned* alpha, int max_bits){}
template<>
void caffe_cpu_quantizec<float>(const int count, const float* X, float* Y, float* alpha, int max_bits){
	//首先找到最大值，进行缩放，缩放到[-127,127]之间（-128不能取到）
	float max_n = fabsf(X[0]);
	int mindex=0;
	//用cblas求最大绝对值
	mindex=cblas_isamax(count,X,1);
	//LOG(INFO)<<fabsf(X[mindex+1])<<","<<fabsf(X[mindex])<<","<<fabsf(X[mindex-1]);
	max_n=fabsf(X[mindex]);
	//scale:scaler=127/max_n;
	float dst_range=(float)pow(2,max_bits-1)-1;
	float scaler=dst_range/max_n;
	*alpha=1/scaler;//缩放因子
	caffe_cpu_scale(count, scaler, X, Y);
	//量化为整型
	for (int i = 0; i < count; i++){
		Y[i] = (Y[i] > 0.0) ? floor(Y[i] + 0.5) : ceil(Y[i] - 0.5);
	}
}
template<>
void caffe_cpu_quantizec<double>(const int count, const double* X, double* Y, double* alpha, int max_bits){
	//首先找到最大值，进行缩放，缩放到[-127,127]之间（-128不能取到）
	int mindex=cblas_idamax(count,X,1);
	double max_n = fabsf(X[mindex]);
	//scale:scaler=127/max_n;
	double dst_range=(double)pow(2,max_bits-1)-1;
	double scaler=dst_range/max_n;
	*alpha=1/scaler;//缩放因子
	caffe_cpu_scale(count, scaler, X, Y);
	//量化为整型
	for (int i = 0; i < count; i++){
		Y[i] = (Y[i] > 0.0) ? floor(Y[i] + 0.5) : ceil(Y[i] - 0.5);
	}
}
//ulq
//ULQ：Q=Round(      Clip          ((F-delta)*alpha+0.5))#alpha=1/alpha,此处是为了避免除法
//		       [1-2^(k-1),2^(k-1)]
template<>
void caffe_cpu_ulq<int>(const int count, const int mean_old, const int sigma_old,const int* X, int* Y, int max_bits,bool restore){}
template<>
void caffe_cpu_ulq<unsigned>(const int count, const unsigned mean_old, const unsigned sigma_old,const unsigned* X, unsigned* Y, int max_bits,bool restore){}
template<>
void caffe_cpu_ulq<float>(const int count, const float mean_old, const float sigma_old,const float* X, float* Y, int max_bits,bool restore){
	float mins=1-pow(2,max_bits-1);
	float maxs=pow(2,max_bits-1);
	for(int i=0;i<count;i++){
		//scale
		Y[i]=(X[i]-mean_old)*sigma_old+0.5;
		//clip
		if(Y[i]>maxs){Y[i]=maxs;}
		if(Y[i]<mins){Y[i]=mins;}
		//round
		Y[i]=(Y[i] > 0.0) ? floor(Y[i] + 0.5) : ceil(Y[i] - 0.5);
		if (restore){
		//还原为相同的range，以进行前向传播
			Y[i]=((Y[i]-0.5)/sigma_old)+mean_old;
		}
	}
}
template<>
void caffe_cpu_ulq<double>(const int count, const double mean_old, const double sigma_old,const double* X, double* Y, int max_bits,bool restore){}
//meanstd
template<>
void caffe_cpu_meanstd<int>(const int count, const int* X, int& mean, int& std){}
template<>
void caffe_cpu_meanstd<unsigned>(const int count, const unsigned* X, unsigned& mean, unsigned& std){}
template<>
void caffe_cpu_meanstd<float>(const int count, const float* X, float& mean, float& stds){
	//calc mean
	//float total=cblas_ssum(count, X, 1);
	//LOG(INFO)<<"count="<<count<<",X[0]="<<X[0]<<",X[count-1]="<<X[count-1];
	float total=0.0;
	for(int i=0;i<count;i++){
		total+=X[i];
	}
	mean=total/count;
	//LOG(INFO)<<"CPU sum is "<<total;
	//calc std
	total=caffe_cpu_dot(count, X, X);
	stds=sqrt(total/count-pow(mean,2));
	//LOG(INFO)<<"CPU mean="<<mean<<",std="<<stds;
}
template<>
void caffe_cpu_meanstd<double>(const int count, const double* X, double& mean, double& stds){
	//calc mean
	//double total=cblas_dsum(count, X, 1);
	double total=0.0;
	for(int i=0;i<count;i++){
		total+=X[i];
	}
	mean=total/count;
	//calc std
	total=caffe_cpu_dot(count, X, X);
	stds=sqrt(total/count-pow(mean,2));
	//LOG(INFO)<<"mean="<<mean<<",std="<<stds;
}

}  // namespace caffe
