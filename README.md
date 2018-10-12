
# orb-slam2_with_semantic_label  yolo-v2 语义-slam

# 结合要点!!!!!!
 
    整合orbslam2-pc 和 yolov3-se
[orbslam2-pc](https://github.com/Ewenwan/ORBSLAM2_with_pointcloud_map)

[yolov3-se](https://github.com/Ewenwan/YOLOv3_SpringEdition)

# 相比 orbslam2-pc 变动的地方

> 1. 头文件部分 
```c
  include/gco-v3.0   图割能量最小 software for energy minimization with graph cuts
  include/config.h   添加了一个参数配置类
  include/lsa_tr.h   基于图割方法的图像分割，是一种重要的图像分割个方法  最大流-最小割 MAXFLOW-MINCUT
                     opencv实现了min-cut/max-flow代码，在opencv/sources/modules/imgproc/src文件下边的gcgraph.hpp中。
  include/pointcloudmapping.h    点云建图程序也有所修改    主要融合部分!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  include/segmentation.h
  include/segmentation_helpers.h
```

> config.h
```c
class Config{
public:

        Config():voxel_resolution(0.01f), // 类默认 构造函数  体素格滤波 尺寸大小
        seed_resolution(0.1f),
        color_importance (1.0f),
        spatial_importance (0.4f),
        normal_importance  (1.0f),
        use_single_cam_transform (false),
        use_supervoxel_refinement (false),

        // Default parameters for model fitting
        use_random_sampling (false),
        noise_threshold(0.02f),
        smooth_cost (0.001),
        min_inliers_per_plane (100),
       	min_plane_area(0.025),
        max_num_iterations (25),
        max_curvature (0.01f),
        gc_scale (1e3){}

    public:

    float voxel_resolution;
    float seed_resolution;
    float color_importance;
    float spatial_importance;
    float normal_importance;
    bool use_single_cam_transform;
    bool use_supervoxel_refinement;

    // Default parameters for model fitting
    bool use_random_sampling;
    float noise_threshold;
    float smooth_cost;
    int min_inliers_per_plane;
    float min_plane_area;
    float label_cost;
    int max_num_iterations;
    float max_curvature;
    int gc_scale;


    Config& operator=(const Config& config) // 类 等号赋值操作符重载========
    {

        voxel_resolution=config.voxel_resolution;
        seed_resolution=config.seed_resolution;
        color_importance=config.color_importance;
        spatial_importance=config.spatial_importance;
        normal_importance=config.normal_importance;
        use_single_cam_transform=config.use_single_cam_transform;
        use_supervoxel_refinement=config.use_supervoxel_refinement;

        use_random_sampling=config.use_random_sampling;
        noise_threshold=config.noise_threshold;
        smooth_cost=config.smooth_cost;
        min_inliers_per_plane=config.min_inliers_per_plane;
        min_plane_area=config.min_plane_area;
        label_cost=config.label_cost;
        max_num_iterations=config.max_num_iterations;
        max_curvature=config.max_curvature;
        gc_scale=config.gc_scale;

    }

};

```



> lsa_tr.h 文件算法分析
```c

```

> pointcloudmapping.h 文件修改分析
```c
#include "YOLOv3SE.h" // 增加目标检测类 yolov3检测

#include <pcl/filters/statistical_outlier_removal.h>// 在原有体素格滤波下，添加 统计学滤波 剔除外点
#include <pcl/filters/passthrough.h>                // 直通滤波器，剔除固定范围内的点

// for clustering
#include <pcl/filters/extract_indices.h>  // 根据索引(来自目标检测)提取点云
#include <pcl/ModelCoefficients.h>        // 模型聚类分割
#include <pcl/features/normal_3d.h>       // 法线特征
#include <pcl/kdtree/kdtree.h>            // 二叉树 点云搜索
#include <pcl/sample_consensus/method_types.h> // 相似性 方法
#include <pcl/sample_consensus/model_types.h>  // 相似性 模型
#include <pcl/segmentation/sac_segmentation.h> // 相似性 分割
#include <pcl/segmentation/extract_clusters.h> // 分割提取点云
#include <pcl/segmentation/region_growing.h>   // 距离区域增长算法分割
#include <pcl/io/pcd_io.h>
#include <pcl/search/search.h>                 // 搜索算法
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>// 欧式距离聚类分割算法
#include <pcl/segmentation/region_growing_rgb.h>// 颜色区域增长算法

#include <iostream>

typedef pcl::PointXYZRGBA PointT;              // 点类型
typedef pcl::PointCloud<PointT> PointCloudT;   // 点云类型
typedef pcl::PointNormal PointNT;              // 点法线特征
typedef pcl::PointCloud<PointNT> PointNCloudT; // 带有法线的点云
typedef pcl::PointXYZL PointLT;                // 带有标签label的点
typedef pcl::PointCloud<PointLT> PointLCloudT;    // 带有标签label的点形成的点云
typedef pcl::PointCloud<pcl::PointXYZL> pointcloudL;


class PointCloudMapping
{
public: 
  ...
  enum cloudType
  {RAW = 0, FILTERED = 1, REMOVAL = 2, CLUSTER = 3}; // 增加点云类型的一个枚举变量
  
  std::vector<cv::Scalar> colors;// 图像颜色，图像个通道值。。。。用处大大滴
  
// 增添 一大波函数 ...

    void final_process(); // 最后处理======
//    void filtCloud(float leaveSize = 0.014f);
//    void removePlane();
//    void cluster();

    YOLOv3 detector;      // 目标检测对象======
    cv::Mat dye_gray(cv::Mat &gray);
    PointCloud::Ptr ECE(PointCloud::Ptr cloud);
    PointCloud::Ptr cylinderSeg(PointCloud::Ptr cloud);        // 圆柱体分割
    PointCloud::Ptr regionGrowingSeg(PointCloud::Ptr cloud_in);// 区域增长分割
    PointCloud::Ptr colorRegionGrowingSeg(pcl::PointCloud <pcl::PointXYZRGB>::Ptr  cloud_in);// 颜色区域增长分割

    PointCloud::Ptr CEC(const std::string& filename);
    void PointCloudXYZRGBAtoXYZ(const pcl::PointCloud<pcl::PointXYZRGBA>& in,
                            pcl::PointCloud<pcl::PointXYZ>& out);// xyzrgba 点云 转换成 无颜色的点云 xyz点云
                            
    void PointXYZRGBAtoXYZ(const pcl::PointXYZRGBA& in,
                                pcl::PointXYZ& out);             // xyzrgba点 转换成 xyz点
                                
    void PointXYZRGBtoXYZRGBA(const pcl::PointXYZRGB& in,
                                pcl::PointXYZRGBA& out);         // xyzrgb点 转换成 xyzrgba点
                                
    void PointCloudXYZRGBtoXYZRGBA(const pcl::PointCloud<pcl::PointXYZRGB>& in,
                            pcl::PointCloud<pcl::PointXYZRGBA>& out);// xyzrgb点云 转换成 xyzrgba点云
                            
    void PointXYZLtoXYZ(const pcl::PointXYZL& in,
                   pcl::PointXYZ& out);                          // xyzl点 转换成 xyz点
                   
    void obj2pcd(const std::string& in, const std::string& out);
    void poisson_reconstruction(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr incloud);// 泊松重构
    void cpf_seg(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr incloud);


  
  
protected:
  ...
  PointLCloudT::Ptr lableMap; // 带有标签的点 形成的点云 地图
  pcl::StatisticalOutlierRemoval<PointT> sor;// 创建统计学滤波器对象

}


```

> segmentation.h 分割类头文件
```c
// 重要头文件
// Graph cuts
#include "gco-v3.0/GCoptimization.h"

// LSA TR Optimisation
#include "lsa_tr.h"

#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/supervoxel_clustering.h>
// 超体聚类 http://www.cnblogs.com/ironstark/p/5013968.html

#include "segmentation_helpers.h"
// Boost
#include <boost/format.hpp>
#include <boost/config.hpp>
#include <boost/graph/adjacency_list.hpp>       // 图 相关算法
#include <boost/graph/connected_components.hpp>
#include <boost/math/special_functions/fpclassify.hpp>

typedef pcl::PointXYZRGBA PointT;  // The point type used for input

    class Segmentation{
    public:
        ... 
        void doSegmentation(); // 分割算法!!!!!!!!!!!!!!!!!!!!!!!!!!
        pcl::PointCloud<pcl::PointXYZL>::Ptr getSegmentedPointCloud(){return segmented_cloud_ptr_;}// 
    private:
        Config config_;
        pcl::PointCloud<PointT>::Ptr input_cloud_ptr_; // 输入点云为 普通的xyzrgba点云
        pcl::PointCloud<pcl::PointXYZL>::Ptr segmented_cloud_ptr_;// 分割算法输出类 输出点云 为xyzl点云，带标签

```




**Authors:** Xuxiang Qi(qixuxiang16@nudt.edu.cn),Shaowu Yang(shaowu.yang@nudt.edu.cn),Yuejin Yan(nudtyyj@nudt.edu.cn)

**Current version:** 1.0.0

* Note: This repository is mainly built upon [ORB_SLAM2](https://github.com/raulmur/ORB_SLAM2) and [YOLO](https://github.com/pjreddie/darknet/). Many thanks for their great work.

## introduction

**orb-slam2_with_semantic_label** is a  visual SLAM system based on  **[ORB_SLAM2[1-2]](https://github.com/raulmur/ORB_SLAM2)**.
The ORB-SLAM2 is a great visual SLAM method that has been popularly applied in  robot applications. However, this method cannot provide semantic information in environmental mapping.In this work,we present a method to build a 3D dense semantic map,which utilize both 2D image labels from **[YOLOv3[3]](https://github.com/qixuxiang/YOLOv3_SpringEdition)** and 3D geometric information.




## 0. Related Publications

**coming soon...**

## 1. Prerequisites

### 1.0 requirements
  * Ubuntu 14.04/Ubuntu 16.04
  * ORBSLAM2 
  * CUDA >=6.5
  * C++11(must)
  * gcc5(must)
  * cmake


### 1.1 Installation

Refer to the corresponding original repositories ([ORB_SLAM2](https://github.com/raulmur/ORB_SLAM2) and [YOLO](https://github.com/qixuxiang/YOLOv3_SpringEdition) for installation tutorial).



### 2.1 Build 

You should follow the instructions provided by ORB_SLAM2 build its dependencies, we do not list here.
You also need to install NVIDIA and cuda to accelerate it.


### 2.2 run 
1. Download  yolov3.weights, yolov3.cfg and coco.names and put them to bin folder,they can be found in [YOLO V3](https://github.com/qixuxiang/YOLOv3_SpringEdition).

2. Download a sequence from http://vision.in.tum.de/data/datasets/rgbd-dataset/download and uncompress it to data folder.
3. Associate RGB images and depth images using the python script [associate.py](http://vision.in.tum.de/data/datasets/rgbd-dataset/tools). We already provide associations for some of the sequences in *Examples/RGB-D/associations/*. You can generate your own associations file executing:

  ```
  python associate.py PATH_TO_SEQUENCE/rgb.txt PATH_TO_SEQUENCE/depth.txt > associations.txt
  ```


4. Execute the following c Change `TUMX.yaml` to TUM1.yaml,TUM2.yaml or TUM3.yaml for freiburg1, freiburg2 and freiburg3 sequences respectively. Change `PATH_TO_SEQUENCE_FOLDER`to the uncompressed sequence folder.My command is :

```
cd bin
./rgbd_tum ../Vocabulary/ORBvoc.txt ../Examples/RGB-D/TUM2.yaml ../data/rgbd-data ../data/rgbd-data/associations.txt

```

## Reference
[1] Mur-Artal R, Montiel J M M, Tardos J D. ORB-SLAM: a versatile and accurate monocular SLAM system[J]. IEEE Transactions on Robotics, 2015, 31(5): 1147-1163.

[2] Mur-Artal R, Tardos J D. ORB-SLAM2: an Open-Source SLAM System for Monocular, Stereo and RGB-D Cameras[J]. arXiv preprint arXiv:1610.06475, 2016.

[3] Redmon, Joseph, and A. Farhadi. "YOLOv3: An Incremental Improvement." (2018).

## License
Our system is released under a [GPLv3 license](https://github.com/qixuxiang/orb-slam2_with_semantic_label/blob/master/License-gpl.txt).

If you want to use code for commercial purposes, please contact the authors.

## Other issue
we do not test the code there on ROS bridge/node.The system relies on an extremely fast and tight coupling between the mapping and tracking on the GPU, which I don't believe ROS supports natively in terms of message passing.

We provide a [video](http://v.youku.com/v_show/id_XMzYyOTMyODM2OA==.html?spm=a2h3j.8428770.3416059.1) here.
