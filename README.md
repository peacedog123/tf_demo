# tf_demo
tensorflow prediction demo with c++


主要参照了下面的项目：
https://github.com/cjweeks/tensorflow-cmake

tensorflow_cc和tensorflow_framework是自己从源码编译的，但是运行时需要Eigen和protobuf的支持（包括版本），且这里不想和系统的版本相冲突，故使用了external的方式引入依赖
