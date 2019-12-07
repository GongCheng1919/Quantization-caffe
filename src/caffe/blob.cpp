#include <climits>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void Blob<Dtype>::Reshape(const int num, const int channels, const int height,
    const int width) {
  vector<int> shape(4);
  shape[0] = num;
  shape[1] = channels;
  shape[2] = height;
  shape[3] = width;
  Reshape(shape);
}

template <typename Dtype>
void Blob<Dtype>::Reshape(const vector<int>& shape) {
  CHECK_LE(shape.size(), kMaxBlobAxes);
  count_ = 1;
  shape_.resize(shape.size());
  if (!shape_data_ || shape_data_->size() < shape.size() * sizeof(int)) {
    shape_data_.reset(new SyncedMemory(shape.size() * sizeof(int)));
  }
  int* shape_data = static_cast<int*>(shape_data_->mutable_cpu_data());
  for (int i = 0; i < shape.size(); ++i) {
    CHECK_GE(shape[i], 0);
    if (count_ != 0) {
      CHECK_LE(shape[i], INT_MAX / count_) << "blob size exceeds INT_MAX";
    }
    count_ *= shape[i];
    shape_[i] = shape[i];
    shape_data[i] = shape[i];
  }
  if (count_ > capacity_) {
    capacity_ = count_;
    data_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
    diff_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
	//GC
	//ternary_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
	quantize_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
	//--
  }
}

template <typename Dtype>
void Blob<Dtype>::Reshape(const BlobShape& shape) {
  CHECK_LE(shape.dim_size(), kMaxBlobAxes);
  vector<int> shape_vec(shape.dim_size());
  for (int i = 0; i < shape.dim_size(); ++i) {
    shape_vec[i] = shape.dim(i);
  }
  Reshape(shape_vec);
}

template <typename Dtype>
void Blob<Dtype>::ReshapeLike(const Blob<Dtype>& other) {
  Reshape(other.shape());
}

template <typename Dtype>
Blob<Dtype>::Blob(const int num, const int channels, const int height,
    const int width)
  // capacity_ must be initialized before calling Reshape
  : capacity_(0) {
  Reshape(num, channels, height, width);
  //设置data与ternary切换函数
  //LOG(INFO)<<"EXCHANGE_DATA_TERNARY_ is "<<EXCHANGE_DATA_TERNARY_<<" and init to false";
	EXCHANGE_DATA_TERNARY_=false;
	delta_=Dtype(0);
	alpha_=Dtype(0);
	EXCHANGE_DATA_QUANTIZE_=false;
	fixedpos_=0;
	maxbits_=0;
}

template <typename Dtype>
Blob<Dtype>::Blob(const vector<int>& shape)
  // capacity_ must be initialized before calling Reshape
  : capacity_(0) {
  Reshape(shape);
  //设置data与ternary切换函数
  //LOG(INFO)<<"EXCHANGE_DATA_TERNARY_ is "<<EXCHANGE_DATA_TERNARY_<<" and init to false";
  EXCHANGE_DATA_TERNARY_=false;
  	delta_=Dtype(0);
	alpha_=Dtype(0);
	EXCHANGE_DATA_QUANTIZE_=false;
	fixedpos_=0;
	maxbits_=0;
}

template <typename Dtype>
const int* Blob<Dtype>::gpu_shape() const {
  CHECK(shape_data_);
  return (const int*)shape_data_->gpu_data();
}

template <typename Dtype>
const Dtype* Blob<Dtype>::cpu_data() const {
	//切换三值与全精度值
	if(EXCHANGE_DATA_TERNARY_){
		//LOG(INFO)<<"cpu_data::use exchange func"<<std::endl;
		return cpu_ternary();
	}
	//切换量化与全精度值
	if(EXCHANGE_DATA_QUANTIZE_){
		//LOG(INFO)<<"cpu_data::use exchange func"<<std::endl;
		return cpu_quantize();
	}
  CHECK(data_);
  return (const Dtype*)data_->cpu_data();
}

template <typename Dtype>
void Blob<Dtype>::set_cpu_data(Dtype* data) {
  CHECK(data);
  // Make sure CPU and GPU sizes remain equal
  size_t size = count_ * sizeof(Dtype);
  if (data_->size() != size) {
    data_.reset(new SyncedMemory(size));
    diff_.reset(new SyncedMemory(size));
	//ternary_.reset(new SyncedMemory(size));
  }
  data_->set_cpu_data(data);
}
//GC
template <typename Dtype>
const Dtype* Blob<Dtype>::cpu_ternary() const {
	if (!ternary_){
		return NULL;
	}
  CHECK(ternary_);
  return (const Dtype*)ternary_->cpu_data();
}

template <typename Dtype>
void Blob<Dtype>::set_cpu_ternary(Dtype* data) {
  CHECK(data);
  // Make sure CPU and GPU sizes remain equal
  size_t size = count_ * sizeof(Dtype);
  if (data_->size() != size) {
    //data_.reset(new SyncedMemory(size));
    //diff_.reset(new SyncedMemory(size));
	ternary_.reset(new SyncedMemory(size));
  }
  ternary_->set_cpu_data(data);
}
template <typename Dtype>
const Dtype* Blob<Dtype>::gpu_ternary() const {
	if (!ternary_){
		return NULL;
	}
  CHECK(ternary_);
  return (const Dtype*)ternary_->gpu_data();
}

template <typename Dtype>
void Blob<Dtype>::set_gpu_ternary(Dtype* data) {
  CHECK(data);
  // Make sure CPU and GPU sizes remain equal
  size_t size = count_ * sizeof(Dtype);
  if (data_->size() != size) {
    //data_.reset(new SyncedMemory(size));
    //diff_.reset(new SyncedMemory(size));
	ternary_.reset(new SyncedMemory(size));
  }
  ternary_->set_gpu_data(data);
}
template <typename Dtype>
Dtype* Blob<Dtype>::mutable_cpu_ternary() {
	if (!ternary_){
		return NULL;
	}
  CHECK(ternary_);
  return static_cast<Dtype*>(ternary_->mutable_cpu_data());
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_gpu_ternary() {
	if (!ternary_){
		return NULL;
	}
  CHECK(ternary_);
  return static_cast<Dtype*>(ternary_->mutable_gpu_data());
}
//quantize
template <typename Dtype>
const Dtype* Blob<Dtype>::cpu_quantize() const {
	if (!quantize_){
		return NULL;
	}
  CHECK(quantize_);
  return (const Dtype*)quantize_->cpu_data();
}

template <typename Dtype>
void Blob<Dtype>::set_cpu_quantize(Dtype* data) {
  CHECK(data);
  // Make sure CPU and GPU sizes remain equal
  size_t size = count_ * sizeof(Dtype);
  if (data_->size() != size) {
    data_.reset(new SyncedMemory(size));
    diff_.reset(new SyncedMemory(size));
	quantize_.reset(new SyncedMemory(size));
  }
  quantize_->set_cpu_data(data);
}
template <typename Dtype>
const Dtype* Blob<Dtype>::gpu_quantize() const {
	if (!quantize_){
		return NULL;
	}
  CHECK(quantize_);
  return (const Dtype*)quantize_->gpu_data();
}

template <typename Dtype>
void Blob<Dtype>::set_gpu_quantize(Dtype* data) {
  CHECK(data);
  // Make sure CPU and GPU sizes remain equal
  size_t size = count_ * sizeof(Dtype);
  if (data_->size() != size) {
    data_.reset(new SyncedMemory(size));
    diff_.reset(new SyncedMemory(size));
	quantize_.reset(new SyncedMemory(size));
  }
  quantize_->set_gpu_data(data);
}
template <typename Dtype>
Dtype* Blob<Dtype>::mutable_cpu_quantize() {
	if (!quantize_){
		return NULL;
	}
  CHECK(quantize_);
  return static_cast<Dtype*>(quantize_->mutable_cpu_data());
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_gpu_quantize() {
	if (!quantize_){
		return NULL;
	}
  CHECK(quantize_);
  return static_cast<Dtype*>(quantize_->mutable_gpu_data());
}
//交换data_与ternary_的地址，以实现前向使用ternary，反向使用data
template <typename Dtype>
void Blob<Dtype>::exchange_data_ternary(bool flag){
	EXCHANGE_DATA_TERNARY_=flag;
  }
  //交换data_与quantize_的地址，以实现前向使用quantize_，反向使用data
template <typename Dtype>
void Blob<Dtype>::exchange_data_quantize(bool flag){
	EXCHANGE_DATA_QUANTIZE_=flag;
  }
//--

template <typename Dtype>
const Dtype* Blob<Dtype>::gpu_data() const {
  	//切换三值与全精度值
	if(EXCHANGE_DATA_TERNARY_){
		//LOG(INFO)<<"gpu_data:: exchange func"<<std::endl;
		return gpu_ternary();
	}
	//切换量化与全精度值
	if(EXCHANGE_DATA_QUANTIZE_){
		//LOG(INFO)<<"cpu_data::use exchange func"<<std::endl;
		return gpu_quantize();
	}
  CHECK(data_);
  return (const Dtype*)data_->gpu_data();
}

template <typename Dtype>
void Blob<Dtype>::set_gpu_data(Dtype* data) {
  CHECK(data);
  // Make sure CPU and GPU sizes remain equal
  size_t size = count_ * sizeof(Dtype);
  if (data_->size() != size) {
    data_.reset(new SyncedMemory(size));
    diff_.reset(new SyncedMemory(size));
	ternary_.reset(new SyncedMemory(size));
  }
  data_->set_gpu_data(data);
}

template <typename Dtype>
const Dtype* Blob<Dtype>::cpu_diff() const {
  CHECK(diff_);
  return (const Dtype*)diff_->cpu_data();
}

template <typename Dtype>
const Dtype* Blob<Dtype>::gpu_diff() const {
  CHECK(diff_);
  return (const Dtype*)diff_->gpu_data();
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_cpu_data() {
  CHECK(data_);
  return static_cast<Dtype*>(data_->mutable_cpu_data());
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_gpu_data() {
  CHECK(data_);
  return static_cast<Dtype*>(data_->mutable_gpu_data());
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_cpu_diff() {
  CHECK(diff_);
  return static_cast<Dtype*>(diff_->mutable_cpu_data());
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_gpu_diff() {
  CHECK(diff_);
  return static_cast<Dtype*>(diff_->mutable_gpu_data());
}

template <typename Dtype>
void Blob<Dtype>::ShareData(const Blob& other) {
  CHECK_EQ(count_, other.count());
  data_ = other.data();
  ternary_=other.ternary();
  quantize_=other.quantize();
}

template <typename Dtype>
void Blob<Dtype>::ShareDiff(const Blob& other) {
  CHECK_EQ(count_, other.count());
  diff_ = other.diff();
}

// The "update" method is used for parameter blobs in a Net, which are stored
// as Blob<float> or Blob<double> -- hence we do not define it for
// Blob<int> or Blob<unsigned int>.
template <> void Blob<unsigned int>::Update() { NOT_IMPLEMENTED; }
template <> void Blob<int>::Update() { NOT_IMPLEMENTED; }

template <typename Dtype>
void Blob<Dtype>::Update() {
  // We will perform update based on where the data is located.
  switch (data_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
  
    // perform computation on CPU
    caffe_axpy<Dtype>(count_, Dtype(-1),
        static_cast<const Dtype*>(diff_->cpu_data()),
        static_cast<Dtype*>(data_->mutable_cpu_data()));
    break;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    // perform computation on GPU
    caffe_gpu_axpy<Dtype>(count_, Dtype(-1),
        static_cast<const Dtype*>(diff_->gpu_data()),
        static_cast<Dtype*>(data_->mutable_gpu_data()));
#else
    NO_GPU;
#endif
    break;
  default:
    LOG(FATAL) << "Syncedmem not initialized.";
  }
}

template <> unsigned int Blob<unsigned int>::asum_data() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <> int Blob<int>::asum_data() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <typename Dtype>
Dtype Blob<Dtype>::asum_data() const {
  if (!data_) { return 0; }
  switch (data_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
  
    return caffe_cpu_asum(count_, cpu_data());
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
  {
    Dtype asum;
    caffe_gpu_asum(count_, gpu_data(), &asum);
    return asum;
  }
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return 0;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
  }
  return 0;
}

template <> unsigned int Blob<unsigned int>::asum_diff() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <> int Blob<int>::asum_diff() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <typename Dtype>
Dtype Blob<Dtype>::asum_diff() const {
  if (!diff_) { return 0; }
  switch (diff_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    return caffe_cpu_asum(count_, cpu_diff());
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
  {
    Dtype asum;
    caffe_gpu_asum(count_, gpu_diff(), &asum);
    return asum;
  }
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return 0;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << diff_->head();
  }
  return 0;
}

template <> unsigned int Blob<unsigned int>::sumsq_data() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <> int Blob<int>::sumsq_data() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <typename Dtype>
Dtype Blob<Dtype>::sumsq_data() const {
  Dtype sumsq;
  const Dtype* data;
  if (!data_) { return 0; }
  switch (data_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
  
    data = cpu_data();
    sumsq = caffe_cpu_dot(count_, data, data);
    break;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    data = gpu_data();
    caffe_gpu_dot(count_, data, data, &sumsq);
#else
    NO_GPU;
#endif
    break;
  case SyncedMemory::UNINITIALIZED:
    return 0;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
  }
  return sumsq;
}

template <> unsigned int Blob<unsigned int>::sumsq_diff() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <> int Blob<int>::sumsq_diff() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <typename Dtype>
Dtype Blob<Dtype>::sumsq_diff() const {
  Dtype sumsq;
  const Dtype* diff;
  if (!diff_) { return 0; }
  switch (diff_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    diff = cpu_diff();
    sumsq = caffe_cpu_dot(count_, diff, diff);
    break;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    diff = gpu_diff();
    caffe_gpu_dot(count_, diff, diff, &sumsq);
    break;
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return 0;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
  }
  return sumsq;
}

template <> void Blob<unsigned int>::scale_data(unsigned int scale_factor) {
  NOT_IMPLEMENTED;
}

template <> void Blob<int>::scale_data(int scale_factor) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void Blob<Dtype>::scale_data(Dtype scale_factor) {
  Dtype* data;
  if (!data_) { return; }
  switch (data_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
  
    data = mutable_cpu_data();
    caffe_scal(count_, scale_factor, data);
    return;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    data = mutable_gpu_data();
    caffe_gpu_scal(count_, scale_factor, data);
    return;
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
  }
}

template <> void Blob<unsigned int>::scale_diff(unsigned int scale_factor) {
  NOT_IMPLEMENTED;
}

template <> void Blob<int>::scale_diff(int scale_factor) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void Blob<Dtype>::scale_diff(Dtype scale_factor) {
  Dtype* diff;
  if (!diff_) { return; }
  switch (diff_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    diff = mutable_cpu_diff();
    caffe_scal(count_, scale_factor, diff);
    return;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    diff = mutable_gpu_diff();
    caffe_gpu_scal(count_, scale_factor, diff);
    return;
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << diff_->head();
  }
}

template <typename Dtype>
bool Blob<Dtype>::ShapeEquals(const BlobProto& other) {
  if (other.has_num() || other.has_channels() ||
      other.has_height() || other.has_width()) {
    // Using deprecated 4D Blob dimensions --
    // shape is (num, channels, height, width).
    // Note: we do not use the normal Blob::num(), Blob::channels(), etc.
    // methods as these index from the beginning of the blob shape, where legacy
    // parameter blobs were indexed from the end of the blob shape (e.g., bias
    // Blob shape (1 x 1 x 1 x N), IP layer weight Blob shape (1 x 1 x M x N)).
    return shape_.size() <= 4 &&
           LegacyShape(-4) == other.num() &&
           LegacyShape(-3) == other.channels() &&
           LegacyShape(-2) == other.height() &&
           LegacyShape(-1) == other.width();
  }
  vector<int> other_shape(other.shape().dim_size());
  for (int i = 0; i < other.shape().dim_size(); ++i) {
    other_shape[i] = other.shape().dim(i);
  }
  return shape_ == other_shape;
}

//@ 2018-8-28
template <typename Dtype>
float Blob<Dtype>::get_alpha() const{
  return alpha_;
}
template <typename Dtype>
float Blob<Dtype>::get_delta() const{
  return delta_;
}
template <typename Dtype>
int Blob<Dtype>::get_fixedpos() const{
  return fixedpos_;
}
template <typename Dtype>
int Blob<Dtype>::get_maxbits() const{
  return maxbits_;
}
//每次更新之后，由权值更新alpha和delta,其中会先计算delta，然后对权值三值化，然后计算alpha
template <typename Dtype>
bool Blob<Dtype>::set_delta(){
  //1、计算delta
  float TERNARY_DELTA=7.0;
  float scale_factor = TERNARY_DELTA * 1.0 / 10; 
  Dtype delta = (Dtype) scale_factor * this->asum_data() / this->count();
  delta = (delta <= 100) ? delta : 100;
  delta = (delta >= -100) ? delta : -100; 
  this->delta_ = delta;
  return true;
}
template <typename Dtype>
bool Blob<Dtype>::set_alpha(const Dtype alpha){
	alpha_=alpha;
	return true;
}
template <typename Dtype>
bool Blob<Dtype>::set_maxbits(const int maxbits){
	maxbits_=maxbits;
	return true;
}
// 三值化
// revised 2016-3-21
template <typename Dtype>
void Blob<Dtype>::quantize_data(Phase phase,CompressParameter compress_param,string compress_type){
	if(phase==TEST && quantize_ && compress_type=="weights"){//如果是weights，则不需要重新根据全精度值计算量化值
		//LOG(INFO)<<"quantize_="<<quantize_<<",compress_type="<<compress_type
		//<<",fixedpos_="<<fixedpos_<<",maxbits_="<<maxbits_<<std::endl;
		return;
	}
	//权值量化，则记录较多
  // const Dtype delta = 0; // default value; 
  // const Dtype delta = (Dtype) 0.8 * this->asum_data() / this->count();
  //int fixed_pos=0;
  int max_bits=8;
  bool calc_fixed_point=true;
  if(phase == TEST){
	  //LOG(INFO)<<"Quantize phase==Test : has_fixedpos="<<compress_param.has_fixedpos()
		//	<<",fixedpos="<<compress_param.fixedpos()<<std::endl;
	  if(compress_param.has_fixedpos() && compress_param.fixedpos()!=0){
		fixedpos_=compress_param.fixedpos();
		calc_fixed_point=false;
	  }
  }
  if(compress_param.has_maxbits() && compress_param.maxbits()!=0){
	maxbits_=compress_param.maxbits();
  }else{
	  set_maxbits(max_bits);
  }
	if(compress_type=="activations"){
		bool scale_back=true;
		if (!data_) { return; }
		switch (data_->head()) {
			case SyncedMemory::HEAD_AT_CPU:
			{
				//LOG(INFO)<<"Activations From CPU";
				caffe_cpu_quantizea<Dtype>(this->count(), (const Dtype*)data_->cpu_data(), this->mutable_cpu_data(),&fixedpos_,maxbits_,true);
				//缩放
				if(scale_back){
					Dtype scaler=pow(2,fixedpos_);
					caffe_cpu_scale(this->count(), scaler, this->cpu_data(), this->mutable_cpu_data());
				}
			}
			return;
			case SyncedMemory::HEAD_AT_GPU:
			case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
			{
				//LOG(INFO)<<"Activations From GPU";
				caffe_gpu_quantizea<Dtype>(this->count(), (const Dtype*)data_->gpu_data(), this->mutable_gpu_data(),&fixedpos_,maxbits_,true);
				//缩放
				if(scale_back){
					Dtype scaler=pow(2,fixedpos_);
					caffe_gpu_scale(this->count(), scaler, this->gpu_data(), this->mutable_gpu_data());
				}
			}
			return;
#else
			NO_GPU;
#endif
			case SyncedMemory::UNINITIALIZED:
			return;
			default:
			LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
		}
		return;
	}
	//LOG(INFO)<<"Quantize data function : calc_fixed_point="<<calc_fixed_point
	//		<<", fixedpos_="<<fixedpos_
	//		<<", maxbits_="<<maxbits_
	//		<<std::endl;
	else if(compress_type=="weights"){
	  if(maxbits_<=0){return;}
	  if (!data_) { return; }
	  CHECK(data_);
	  //LOG(INFO)<<"head="<<data_->head()<<",SyncedMemory::HEAD_AT_CPU="
	  //<<SyncedMemory::HEAD_AT_CPU<<",SyncedMemory::SYNCED="
	  //<<SyncedMemory::SYNCED;
	  switch (data_->head()) {
	  case SyncedMemory::HEAD_AT_CPU:
		{	
		//LOG(INFO)<<"Go to CPU mode";
		caffe_cpu_quantizea<Dtype>(this->count(), (const Dtype*)data_->cpu_data(), this->mutable_cpu_quantize(),&fixedpos_,maxbits_,calc_fixed_point);
		Dtype scaler=pow(2,fixedpos_);
		caffe_cpu_scale(this->count(), scaler, this->cpu_quantize(), this->mutable_cpu_quantize());
		//LOG(INFO)<<"end CPU mode";
		}
		return;
	  case SyncedMemory::HEAD_AT_GPU:
	  case SyncedMemory::SYNCED:
	#ifndef CPU_ONLY
		{
		//LOG(INFO)<<"Go to GPU mode";
		caffe_gpu_quantizea<Dtype>(this->count(), (const Dtype*)data_->gpu_data(), this->mutable_gpu_quantize(),&fixedpos_,maxbits_,calc_fixed_point);
		Dtype scaler=pow(2,fixedpos_);
		caffe_gpu_scale(this->count(), scaler, this->gpu_quantize(), this->mutable_gpu_quantize());
		//LOG(INFO)<<"end GPU mode";
		}
		return;
	#else
		NO_GPU;
	#endif
	  case SyncedMemory::UNINITIALIZED:
		return;
	  default:
		LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
	  }
	}
	else{
		LOG(ERROR)<<"Unknown compress_type!";
	}
}
//
//compress_param:the weights compress params
template <typename Dtype>
void Blob<Dtype>::clip_data(Phase phase,CompressParameter compress_param,string compress_type){
	if(phase==TEST && quantize_ && compress_type=="weights"){//如果是weights，则不需要重新根据全精度值计算量化值
		return;
	}
	int max_bits=8;
	if(compress_param.has_maxbits() && compress_param.maxbits()!=0){
		maxbits_=compress_param.maxbits();
	}else{
		set_maxbits(max_bits);
	}
	//LOG(INFO)<<"Clip alpha="<<alpha<<", max_bits="<<max_bits<<std::endl;
	//activation不需要保留全精度备份以及
	if(compress_type=="activations"){
		bool scale_back=true;
		if (!data_) { return; }
		switch (data_->head()) {
			case SyncedMemory::HEAD_AT_CPU:
			{       
				//进行截断(b/c函数)
				caffe_cpu_quantizec(this->count(), this->cpu_data(),this->mutable_cpu_data(),&this->alpha_, max_bits);
				//缩放还原
				if(scale_back){
					caffe_cpu_scale(this->count(), this->alpha_, this->cpu_data(), this->mutable_cpu_data());
				}
			}
				return;
			case SyncedMemory::HEAD_AT_GPU:
			case SyncedMemory::SYNCED:
			#ifndef CPU_ONLY
			{
				//进行截断
				caffe_gpu_quantizec(this->count(), this->gpu_data(),this->mutable_gpu_data(),&this->alpha_, max_bits);
				//缩放还原
				if(scale_back){
					caffe_gpu_scale(this->count(), this->alpha_, this->gpu_data(), this->mutable_gpu_data());
				}
			}
				return;
			#else
				NO_GPU;
			#endif
			case SyncedMemory::UNINITIALIZED:
				return;
			default:
				LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
		}
		return;
	}
	else if(compress_type=="weights"){
		if (!data_) { return; }
		switch (data_->head()) {
			case SyncedMemory::HEAD_AT_CPU:
			{       
				//进行截断(b/c函数)
				caffe_cpu_quantizec(this->count(), this->cpu_data(),this->mutable_cpu_quantize(),&this->alpha_, max_bits);
				//缩放还原
				caffe_cpu_scale(this->count(), this->alpha_, this->cpu_quantize(), this->mutable_cpu_quantize());
			}
				return;
			case SyncedMemory::HEAD_AT_GPU:
			case SyncedMemory::SYNCED:
			#ifndef CPU_ONLY
			{
				//进行截断
				caffe_gpu_quantizec(this->count(), this->gpu_data(),this->mutable_gpu_quantize(),&this->alpha_, max_bits);
				//缩放还原
				caffe_gpu_scale(this->count(), this->alpha_, this->gpu_quantize(), this->mutable_gpu_quantize());
			}
				return;
			#else
				NO_GPU;
			#endif
			case SyncedMemory::UNINITIALIZED:
				return;
			default:
				LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
		}
	}else{
		LOG(ERROR)<<"Unknown compress type!";
	}
}
// 三值化，并对scaler进行量化
// revised 2018-9-6
template <typename Dtype>
void Blob<Dtype>::ternarize_data(Phase phase,bool quantize_alpha,CompressParameter compress_param,string compress_type){
	if(phase==TEST && quantize_ && compress_type=="weights"){
        //LOG(INFO)<<"ternayr_="<<ternary_<<",compress_type="<<compress_type
		//<<",delta_="<<delta_<<",alpha_="<<alpha_<<std::endl;
		return;
	}
  // const Dtype delta = 0; // default value; 
  // const Dtype delta = (Dtype) 0.8 * this->asum_data() / this->count();
	//如果是激活值的量化，则直接做三值化，不需要还原，也不需要记录alpha之类的
	if(compress_type=="activations"){
		if (!data_) { return; }
		switch (data_->head()) {
			case SyncedMemory::HEAD_AT_CPU:
			{       
				Dtype alpha = 1;
				caffe_cpu_ternary<Dtype>(this->count(), delta_, (const Dtype*)data_->cpu_data(), this->mutable_cpu_data(),&alpha);
			}
			return;
			case SyncedMemory::HEAD_AT_GPU:
			case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
			{
				Dtype alpha=1;
				caffe_gpu_ternary<Dtype>(this->count(), delta_, (const Dtype*)data_->gpu_data(), this->mutable_gpu_data(),&alpha);
			}
			return;
#else
			NO_GPU;
#endif
			case SyncedMemory::UNINITIALIZED:
			return;
			default:
			LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
		}
		return;
	}
	//权值量化，则记录较多
   //Dtype delta,alpha;
   //int max_bits=8, fixedpos=0;
   int maxbits=8;
   bool calc_fixed_point=true;
   if(phase == TEST && compress_param.has_delta() && compress_param.delta()!=0){
        //LOG(INFO)<<"Ternary phase==Test : has_delta="<<compress_param.has_delta()
			//<<",delta="<<compress_param.delta()
			//<<",has_alpha="<<compress_param.has_alpha()
			//<<",alpha="<<compress_param.alpha()
			//<<",has_fixedpos="<<compress_param.has_fixedpos()
			//<<",fixedpos="<<compress_param.fixedpos()
			//<<std::endl;
		//使用存在的delta参数
		if(compress_param.has_delta() && compress_param.delta()!=0){
			delta_=compress_param.delta();
		}
		if(compress_param.has_alpha() && compress_param.alpha()!=0){
			alpha_=compress_param.alpha();
		}
		if(compress_param.has_fixedpos()&&compress_param.fixedpos()!=0){
			//maxbits_=compress_param.maxbits();
				fixedpos_=compress_param.fixedpos();
				calc_fixed_point=false;
				Dtype scaler=pow(2,fixedpos_);
				alpha_=alpha_*scaler;
		}
		//LOG(INFO)<<compress_param.delta()<<" | "<<compress_param.alpha()<<" | "<<compress_param.fixedpos()<<" | "<<compress_param.maxbits()<<" | "<<std::endl;
		//LOG(INFO)<<"Test Phase argn is: delta_="<<delta_<<",alpha_"<<alpha_<<",fixedpos_="<<fixedpos_<<std::endl;
  }else{
	  this->set_delta();
	  //delta = this->get_delta();
  }
  if(compress_param.has_maxbits() && compress_param.maxbits()!=0){
	maxbits_=compress_param.maxbits();
  }else{
	  set_maxbits(maxbits);
  }
    //LOG(INFO)<<"Ternary data function : calc_fixed_point="<<calc_fixed_point
	//		<<", delta_="<<delta_
	//		<<", alpha_="<<alpha_
	//		<<", fixedpos_="<<fixedpos_
	//		<<", maxbits_="<<maxbits_
	//		<<std::endl;
  if (!data_) { return; }
  switch (data_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
  {       
        Dtype alpha = 1;
        caffe_cpu_ternary<Dtype>(this->count(), delta_, (const Dtype*)data_->cpu_data(), this->mutable_cpu_quantize(),&alpha);
        //检查是否需要对alpha做量化
		if(quantize_alpha){
			//对alpha进行量化，定点保存到fixedpos_（因为不可能同时三值化和量化）
			//set_maxbits(maxbits);
			//LOG(INFO)<<"alpha = "<<((Dtype*)&alpha)[0];
			caffe_cpu_quantizea<Dtype>(1, (const Dtype*)&alpha, (Dtype*)&alpha,&fixedpos_,maxbits_,calc_fixed_point);
			//LOG(INFO)<<"alpha_q = "<<alpha<<", fixedpos_ = "<<fixedpos_<<"maxbits_="<<maxbits_<<std::endl;
			Dtype scaler=pow(2,fixedpos_);
			alpha=alpha*scaler;//缩放回小数
		}
		if(phase == TRAIN || (phase == TEST && alpha_==0)){
			set_alpha(alpha);
		}
        //
        //缩放应用到权值
        caffe_cpu_scale(this->count(), alpha, this->cpu_quantize(), this->mutable_cpu_quantize());
}
    return;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
{
        Dtype alpha=1;
		caffe_gpu_ternary<Dtype>(this->count(), delta_, (const Dtype*)data_->gpu_data(), this->mutable_gpu_quantize(),&alpha);
            //检查是否需要对alpha做量化
		if(quantize_alpha){
			//对alpha进行量化，定点保存到fixedpos_（因为不可能同时三值化和量化）
			//set_maxbits(maxbits);
			//LOG(INFO)<<"alpha = "<<((Dtype*)&alpha)[0];
			caffe_cpu_quantizea<Dtype>(1, (const Dtype*)&alpha, (Dtype*)&alpha,&fixedpos_,maxbits_,calc_fixed_point);
			//LOG(INFO)<<",alpha_q = "<<alpha<<", fixedpos_ = "<<fixedpos_<<",maxbits_="<<maxbits_<<std::endl;
			Dtype scaler=pow(2,fixedpos_);
			alpha=alpha*scaler;//缩放回小数
		}
        if(phase == TRAIN || (phase == TEST && alpha_==0)){
			set_alpha(alpha);
		}
  //LOG(INFO)<<"delta = "<<delta<<", alpha = "<<alpha<<std::endl;
        caffe_gpu_scale(this->count(), alpha, this->gpu_quantize(), this->mutable_gpu_quantize());
}
    return;
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
  }
}

//ULQ：Q=Round(      Clip          ((F-delta)/alpha+0.5))#alpha=1/alpha,此处是为了避免除法
//		       [1-2^(k-1),2^(k-1)]
template <typename Dtype>
void Blob<Dtype>:: ulq_weights(Phase phase,CompressParameter compress_param){
	if(phase==TEST && quantize_){//如果是weights，则不需要重新根据全精度值计算量化值
		//LOG(INFO)<<"quantize_="<<quantize_<<",compress_type="<<compress_type
		//<<",fixedpos_="<<fixedpos_<<",maxbits_="<<maxbits_<<std::endl;
		return;
	}
	//LOG(INFO)<<"get into ulq_weights function";
	const static double lambdalist[]={1.5950161625215948, 0.99529008541347519, 0.5872211503939504, 0.3358904980253396, 0.18944224088629152, 0.10570877041455067, 0.057082736023857356, 0.030482181036739827};
	//估计计算情况
	int max_bits=8;
	if(compress_param.has_maxbits() && compress_param.maxbits()!=0){
		maxbits_=compress_param.maxbits();
	}else{
		set_maxbits(max_bits);
	}
	//alpha记录scale，delta记录shift，maxbits记录量化到的位宽，fixedpos无用
	if(maxbits_<=0){return;}
	if (!data_) { return; }
	CHECK(data_);
	switch (data_->head()) {
		case SyncedMemory::HEAD_AT_CPU:
		{	//LOG(INFO)<<"Go to CPU mode"<<std::endl;
			//得到压缩后的权值
				//计算alpha_ 和 delta_(均值和标准差)
			caffe_cpu_meanstd(this->count(), this->cpu_data(), this->delta_, this->alpha_);
			Dtype lambdak=Dtype(0);
			//Dtype ups=Dtype(pow(2,this->maxbits_-1));
			//Dtype downs=1-ups;
			//找最大最小值太耗时了，因此此处有3.x sigma代替了（几乎包含了所有的数据）
			if(this->maxbits_>8){
				lambdak=(2*3.1*this->alpha_)/(pow(2,this->maxbits_)-1);
			}else{
				lambdak=Dtype(lambdalist[this->maxbits_-1]);
			}
			this->alpha_=lambdak*this->alpha_+1e-7;
			//算法
			caffe_cpu_ulq(this->count(), this->delta_, 1/this->alpha_, this->cpu_data(), this->mutable_cpu_quantize(), maxbits_);
		}
		return;
		case SyncedMemory::HEAD_AT_GPU:
		case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
		{	//LOG(INFO)<<"Go to GPU mode"<<std::endl;
			caffe_gpu_meanstd(this->count(), this->gpu_data(), this->delta_, this->alpha_);
			//caffe_cpu_meanstd(this->count(), this->cpu_data(), this->delta_, this->alpha_);
			Dtype lambdak=Dtype(0);
			//Dtype ups=Dtype(pow(2,this->maxbits_-1));
			//Dtype downs=1-ups;
			//找最大最小值太耗时了，因此此处有3.x sigma代替了（几乎包含了所有的数据）
			if(this->maxbits_>8){
				lambdak=(2*3.1*this->alpha_)/(pow(2,this->maxbits_)-1);
			}else{
				lambdak=Dtype(lambdalist[this->maxbits_-1]);
			}
			this->alpha_=lambdak*this->alpha_+1e-7;
			//算法
			caffe_gpu_ulq(this->count(), this->delta_, 1/this->alpha_, this->gpu_data(), this->mutable_gpu_quantize(), this->maxbits_);
			//还原权值
		}
		return;
#else
		NO_GPU;
#endif
		case SyncedMemory::UNINITIALIZED:
		return;
		default:
		LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
	}

}
//ULQ：Q=Round(      Clip          ((F-delta)*alpha+0.5))#alpha=1/alpha,此处是为了避免除法
//alpha记录均值分母（不除以N的那部分）之和，delta记录方差分母（不除以N的那部分）之和，fixedpos记录number数，maxbits记录量化到的位宽
//lambda为平均滑动系数
template <typename Dtype>
void Blob<Dtype>:: ulq_activations(Phase phase,CompressParameter compress_param,float lambda){
	float eps=0.0004;
	if(phase==TRAIN){
		//按照CPU/GPU计算均值和标准差
		if (!data_) { return; }
		CHECK(data_);
		Dtype means=0,stds=0;
		switch (data_->head()) {
			case SyncedMemory::HEAD_AT_CPU:
			{	//LOG(INFO)<<"Go to CPU mode"<<std::endl;
				//得到压缩后的权值
					//计算alpha_ 和 delta_(均值和标准差)
				caffe_cpu_meanstd(this->count(), this->cpu_data(), means, stds);
			}
			return;
			case SyncedMemory::HEAD_AT_GPU:
			case SyncedMemory::SYNCED:
	#ifndef CPU_ONLY
			{	//LOG(INFO)<<"Go to GPU mode"<<std::endl;
				caffe_gpu_meanstd(this->count(), this->gpu_data(), means, stds);
			}
			return;
	#else
			NO_GPU;
	#endif
			case SyncedMemory::UNINITIALIZED:
			return;
			default:
			LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
		}
		//计算出当前批的均值和标准差，将其累计到参数上
		compress_param.set_fixedpos(compress_param.fixedpos()+this->count());
		compress_param.set_alpha(compress_param.alpha()+compress_param.fixedpos()*means);
		compress_param.set_delta(compress_param.delta()+pow(stds,2)*compress_param.fixedpos());
	}
	//估计计算情况
	int max_bits=8;
	if(compress_param.has_maxbits() && compress_param.maxbits()!=0){
		maxbits_=compress_param.maxbits();
	}else{
		set_maxbits(max_bits);
	}
	if(maxbits_<=0){return;}
	//计算样本整体的均值和标准差
	this->alpha_=compress_param.alpha()/(eps+compress_param.fixedpos());
	this->delta_=sqrt(compress_param.delta()/(eps+compress_param.fixedpos()));
	//LOG(INFO)<<"get into ulq_weights function";
	const static double lambdalist[]={1.5950161625215948, 0.99529008541347519, 0.5872211503939504, 0.3358904980253396, 0.18944224088629152, 0.10570877041455067, 0.057082736023857356, 0.030482181036739827};
	Dtype lambdak=Dtype(0);
	//找最大最小值太耗时了，因此此处有3.x sigma代替了（几乎包含了所有的数据）
	if(this->maxbits_>8){
		lambdak=(2*3.1*this->alpha_)/(pow(2,this->maxbits_)-1);
	}else{
		lambdak=Dtype(lambdalist[this->maxbits_-1]);
	}
	this->alpha_=lambdak*this->alpha_;
	//对激活值做缩放(不做还原处理)
	if (!data_) { return; }
	CHECK(data_);
	switch (data_->head()) {
		case SyncedMemory::HEAD_AT_CPU:
		{//对激活值做缩放(不做还原处理)
			caffe_cpu_ulq(this->count(), this->delta_, 1/this->alpha_, this->gpu_data(), this->mutable_gpu_quantize(), this->maxbits_,false);
		}
		return;
		case SyncedMemory::HEAD_AT_GPU:
		case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
		{//对激活值做缩放(不做还原处理)
			caffe_gpu_ulq(this->count(), this->delta_, 1/this->alpha_, this->gpu_data(), this->mutable_gpu_quantize(), this->maxbits_,false);
		}
		return;
#else
		NO_GPU;
#endif
		case SyncedMemory::UNINITIALIZED:
		return;
		default:
		LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
	}
}

template <typename Dtype>
void Blob<Dtype>::CopyFrom(const Blob& source, bool copy_diff, bool reshape) {
  if (source.count() != count_ || source.shape() != shape_) {
    if (reshape) {
      ReshapeLike(source);
    } else {
      LOG(FATAL) << "Trying to copy blobs of different sizes.";
    }
  }
  //GC
  delta_=source.get_delta();
  alpha_=source.get_alpha();
  fixedpos_=source.get_fixedpos();
  maxbits_=source.get_maxbits();
  switch (Caffe::mode()) {
  case Caffe::GPU:
    if (copy_diff) {
      caffe_copy(count_, source.gpu_diff(),
          static_cast<Dtype*>(diff_->mutable_gpu_data()));
    } else {
		caffe_copy(count_, source.gpu_data(),
          static_cast<Dtype*>(data_->mutable_gpu_data()));
		//GC
		if(source.gpu_ternary()){
			caffe_copy(count_, source.gpu_ternary(),
			  static_cast<Dtype*>(ternary_->mutable_gpu_data()));
		}
		if(source.gpu_quantize()){
			caffe_copy(count_, source.gpu_quantize(),
			  static_cast<Dtype*>(quantize_->mutable_gpu_data()));
		}
    }
    break;
  case Caffe::CPU:
    if (copy_diff) {
      caffe_copy(count_, source.cpu_diff(),
          static_cast<Dtype*>(diff_->mutable_cpu_data()));
    } else {
		caffe_copy(count_, source.cpu_data(),
          static_cast<Dtype*>(data_->mutable_cpu_data()));
		//GC
		if(source.cpu_ternary()){
			caffe_copy(count_, source.cpu_ternary(),
			  static_cast<Dtype*>(ternary_->mutable_cpu_data()));
		}
		if(source.cpu_quantize()){
			caffe_copy(count_, source.cpu_quantize(),
			  static_cast<Dtype*>(quantize_->mutable_cpu_data()));
		}
    }
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode.";
  }
}

template <typename Dtype>
void Blob<Dtype>::FromProto(const BlobProto& proto, bool reshape,bool FPfirst) {
  if (reshape) {
    vector<int> shape;
    if (proto.has_num() || proto.has_channels() ||
        proto.has_height() || proto.has_width()) {
      // Using deprecated 4D Blob dimensions --
      // shape is (num, channels, height, width).
      shape.resize(4);
      shape[0] = proto.num();
      shape[1] = proto.channels();
      shape[2] = proto.height();
      shape[3] = proto.width();
    } else { 
      shape.resize(proto.shape().dim_size());
      for (int i = 0; i < proto.shape().dim_size(); ++i) {
        shape[i] = proto.shape().dim(i);
      }
    }
    Reshape(shape);
  } else {
    CHECK(ShapeEquals(proto)) << "shape mismatch (reshape not set)";
  }
  // copy data
  // diff
  if (proto.double_diff_size() > 0) {
    CHECK_EQ(count_, proto.double_diff_size());
    Dtype* diff_vec = mutable_cpu_diff();
    for (int i = 0; i < count_; ++i) {
      diff_vec[i] = proto.double_diff(i);
    }
  } else if (proto.diff_size() > 0) {
    CHECK_EQ(count_, proto.diff_size());
    Dtype* diff_vec = mutable_cpu_diff();
    for (int i = 0; i < count_; ++i) {
      diff_vec[i] = proto.diff(i);
    }
  }
  //GC
  // get alpha and delta from proto object
  if(proto.has_delta() && proto.has_alpha()){
	delta_=Dtype(proto.delta());
	alpha_=Dtype(proto.alpha());
  }else{
	delta_=Dtype(0);
	alpha_=Dtype(0);
  }
  // get fixedpos_ and maxbits_ from proto object
  if(proto.has_fixedpos() && proto.has_maxbits()){
	fixedpos_=Dtype(proto.fixedpos());
	maxbits_=Dtype(proto.maxbits());
	//如果alpha！=0,则对alpha做反量化
	//alpha_=alpha_*pow(2,fixedpos_);
  }else{
	fixedpos_=0;
	maxbits_=8;
  }
  //LOG(INFO)<<"delta_="<<delta_<<",alpha_="<<alpha_<<",fixedpos_="<<fixedpos_
  //<<",maxbits_="<<maxbits_<<std::endl;
  //IF FPfirst is true: first copy the float weights, and then, if there are quantized
  //weights, copy them to quantize
  //ELSE: first copy quantized weights to the FP position
  Dtype* data_vec = mutable_cpu_data();
  if (FPfirst){
	  if (proto.double_data_size() > 0) {
		CHECK_EQ(count_, proto.double_data_size());
		for (int i = 0; i < count_; ++i) {
		  data_vec[i] = proto.double_data(i);
		}
	  } else {
		CHECK_EQ(count_, proto.data_size());
		for (int i = 0; i < count_; ++i) {
		  data_vec[i] = proto.data(i);
		}
	  }
	  //generate quantize
	  if (proto.double_quantize_size() > 0) {
		CHECK_EQ(count_, proto.double_quantize_size());
		Dtype* quantize_vec = mutable_cpu_quantize();
		for (int i = 0; i < count_; ++i) {
			//进行反量化
		  //quantize_vec[i] = proto.double_quantize(i)*pow(2,fixedpos_);
		  quantize_vec[i] = proto.double_quantize(i);
		}
	  } else if (proto.quantize_size() > 0) {
		CHECK_EQ(count_, proto.quantize_size());
		Dtype* quantize_vec = mutable_cpu_quantize();
		for (int i = 0; i < count_; ++i) {
			//进行反量化
		  //quantize_vec[i] = proto.quantize(i)*pow(2,fixedpos_);
		  quantize_vec[i] = proto.quantize(i);
		}
	  }	  //generate ternary and save to quantize
	  else if (proto.double_ternary_size() > 0) {
		CHECK_EQ(count_, proto.double_ternary_size());
		Dtype* ternary_vec = mutable_cpu_quantize();
		for (int i = 0; i < count_; ++i) {
			//需要注意的是，会进行scaler缩放
		  //ternary_vec[i] = proto.double_ternary(i)*alpha_;
		  ternary_vec[i] = proto.double_ternary(i);
		}
	  } else if (proto.ternary_size() > 0) {
		CHECK_EQ(count_, proto.ternary_size());
		Dtype* ternary_vec = mutable_cpu_quantize();
		for (int i = 0; i < count_; ++i) {
			//需要注意的是，会进行scaler缩放
		  //ternary_vec[i] = proto.ternary(i)*alpha_;
		  ternary_vec[i] = proto.ternary(i);
		}
	  }
  }else{
	  if (proto.double_quantize_size() > 0){
		  CHECK_EQ(count_, proto.double_quantize_size()); 
		  for (int i = 0; i < count_; ++i) {
			data_vec[i] = proto.double_quantize(i);
		  }
	  }else if (proto.quantize_size() > 0){
		  CHECK_EQ(count_, proto.quantize_size()); 
		  for (int i = 0; i < count_; ++i) {
			data_vec[i] = proto.quantize(i);
		  }
	  }else if(proto.double_ternary_size() > 0){
		  CHECK_EQ(count_, proto.double_ternary_size()); 
		  for (int i = 0; i < count_; ++i) {
			data_vec[i] = proto.double_ternary(i);
		  }
	  }else if (proto.ternary_size() > 0){
		  CHECK_EQ(count_, proto.ternary_size()); 
		  for (int i = 0; i < count_; ++i) {
			data_vec[i] = proto.ternary(i);
		  }
	  }else if (proto.double_data_size() > 0) {
		CHECK_EQ(count_, proto.double_data_size());
		for (int i = 0; i < count_; ++i) {
		  data_vec[i] = proto.double_data(i);
		}
	  }else {
		CHECK_EQ(count_, proto.data_size());
		for (int i = 0; i < count_; ++i) {
		  data_vec[i] = proto.data(i);
		}
	  }
  }
}

template <>
void Blob<double>::ToProto(BlobProto* proto, bool write_diff,bool save_quantize) const {
  proto->clear_shape();
  for (int i = 0; i < shape_.size(); ++i) {
    proto->mutable_shape()->add_dim(shape_[i]);
  }
  proto->clear_double_data();
  proto->clear_double_diff();
  const double* data_vec = cpu_data();
  for (int i = 0; i < count_; ++i) {
    proto->add_double_data(data_vec[i]);
  }
  if (write_diff) {
    const double* diff_vec = cpu_diff();
    for (int i = 0; i < count_; ++i) {
      proto->add_double_diff(diff_vec[i]);
    }
  }
  //检查是不是存在quantize，如果存在，就存quantize，否则不存
  if (!save_quantize){
	return;
  }
  size_t size = count_ * sizeof(double);
  //LOG(INFO)<<size<<"=="<<quantize_->size();
  if(quantize_&&quantize_->size() != size) {
	  LOG(ERROR)<<"Unable to save quantize values since no values!";
	  return;
  }
  proto->clear_double_quantize();
  proto->set_delta(get_delta());
  proto->set_alpha(get_alpha());
  proto->set_fixedpos(get_fixedpos());
  proto->set_maxbits(get_maxbits());
	const double* quantize_vec = cpu_quantize();
	for (int i = 0; i < count_; ++i) {
	  proto->add_double_quantize(quantize_vec[i]);
    }
}

template <>
void Blob<float>::ToProto(BlobProto* proto, bool write_diff,bool save_quantize) const {
  proto->clear_shape();
  for (int i = 0; i < shape_.size(); ++i) {
    proto->mutable_shape()->add_dim(shape_[i]);
  }
  proto->clear_data();
  proto->clear_diff();
  const float* data_vec = cpu_data();
  for (int i = 0; i < count_; ++i) {
    proto->add_data(data_vec[i]);
  }
  if (write_diff) {
    const float* diff_vec = cpu_diff();
    for (int i = 0; i < count_; ++i) {
      proto->add_diff(diff_vec[i]);
    }
  }
    //GC
	//检查是不是存在quantize，如果存在，就存quantize，否则不存
  if (!save_quantize){
	return;
  }
  size_t size = count_ * sizeof(float);
  //LOG(INFO)<<size<<"=="<<quantize_->size();
  if(quantize_&&quantize_->size() != size) {
	  LOG(ERROR)<<"Unable to save quantize values since no values!";
	  return;
  }
  proto->clear_double_quantize();
  proto->set_delta(get_delta());
  proto->set_alpha(get_alpha());
  proto->set_fixedpos(get_fixedpos());
  proto->set_maxbits(get_maxbits());
	const float* quantize_vec = cpu_quantize();
	for (int i = 0; i < count_; ++i) {
	  proto->add_quantize(quantize_vec[i]);
    }
  
}

INSTANTIATE_CLASS(Blob);
template class Blob<int>;
template class Blob<unsigned int>;

}  // namespace caffe

