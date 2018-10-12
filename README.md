
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

> segmentation_helpers.h 分割类头文件 主要判断 两点的凸凹性 用于 点云分割
```c

// 低层次视觉—点云分割（基于凹凸性）=====================================================================================
// 对于二维图像而言，其凹凸性较难描述，但对于三维图像而言，凹凸几乎是与生俱来的性质。
// LCCP是Locally Convex Connected Patches的缩写，翻译成中文叫做 ”局部凸连接打包一波带走:1.基于超体聚类的过分割。2.在超体聚类的基础上再聚类。
// ====== 超体聚类 http://www.cnblogs.com/ironstark/p/5013968.html
// LCCP 凹凸性聚类 https://www.cnblogs.com/ironstark/p/5027269.html 
// LCCP算法在相似物体场景分割方面有着较好的表现，对于颜色类似但棱角分明的物体可使用该算法。（比如X同学仓库里那一堆纸箱）。

// CPC方法的全称为Constrained Planar Cuts，出自论文：Constrained Planar Cuts - Object Partitioning for Point Clouds 。
// 和LCCP方法不同，此方法的分割对象是object。此方法能够将物体分成有意义的块：比如人的肢体等。
// CPC方法可作为AI的前处理，作为RobotVision还是显得有些不合适。
// 但此方法不需要额外训练，自底向上的将三维图像分割 成有明确意义部分，是非常admirable的。
// 和其他基于凹凸性的方法相同，本方法也需要先进行超体聚类。在完成超体聚类之后，采用和LCCP相同的凹凸性判据获得各个块之间的凹凸关系。
// 在获得凹凸性之后，CPC方法所采取的措施是不同的。其操作称为 半全局分割 。
// 在分割之前，首先需要生成 EEC(Euclidean edge cloud)， EEC的想法比较神奇，因为凹凸性定义在相邻两个”片“上，
// 换言之，定义在连接相邻两“片”的edge上。将每个edge抽象成一个点云，则得到了附带凹凸信息的点云。
// 如图所示，左图是普通点云，但附带了邻接和凹凸信息。右边是EEC，对凹边赋权值1，其他为0。 此方法称作  weighted RanSac。
// https://images2015.cnblogs.com/blog/710098/201512/710098-20151208163052980-70483439.jpg
// 显而易见，某处如果蓝色的点多，那么就越 凹，就越应该切开（所谓切开实际上是用平面划分）。
// 问题就转化为利用蓝点求平面了。利用点云求一个最可能的平面当然需要请出我们的老朋友 RanSaC . 
// 但此处引入一个评价函数，用于评价此次分割的 优良程度Sm,Pm 是EEC中的点.


// 在PCL中CPC类继承自 LCCP 类，但是这个继承我觉得不好，这两个类之间并不存在抽象与具体的关系，只是存在某些函数相同而已。
// 不如多设一个 凹凸分割类 作为CPC类与LCCP类的父类，所有的输入接口等都由凹凸分割类提供。
// 由CPC算法和LCCP算法继承凹凸类，作为 凹凸分割 的具体实现。
// 毕竟和 凹凸分割 有关的算法多半是对整体进行分割，和其他点云分割算法区别较大。


// RGB-D Object Dataset ： http://rgbd-dataset.cs.washington.edu/

bool isConvex(Eigen::Vector3f p1, Eigen::Vector3f n1, Eigen::Vector3f p2, Eigen::Vector3f n2, float seed_resolution, float voxel_resolution)
{
    float concavity_tolerance_threshold = 10;
    const Eigen::Vector3f& source_centroid = p1; // 源点云中心
    const Eigen::Vector3f& target_centroid = p2;  // 目标中心

    const Eigen::Vector3f& source_normal = n1;   // 源点云法线
    const Eigen::Vector3f& target_normal = n2;    // 目标点云法线

    if (concavity_tolerance_threshold < 0)		return (false);

    bool is_convex = true;  // 凸???
    bool is_smooth = true;// 平滑

    float normal_angle = pcl::getAngle3D(source_normal, target_normal, true);// 两法线 角度差值 true 表示 返回 度
    //  Geometric comparisons  几何比较 
    Eigen::Vector3f vec_t_to_s, vec_s_to_t;// 中心位置 平移量

    vec_t_to_s = source_centroid - target_centroid;// 目标 到 源 平移量（后面的向量 到 前面的向量）
    vec_s_to_t = -vec_t_to_s;// 源 到 目标 平移量
    
    // 两向量叉积 向量积
    Eigen::Vector3f ncross;
    ncross = source_normal.cross (target_normal);

    // 平滑检测======================================================================
    // Smoothness Check: Check if there is a step between adjacent patches
    bool use_smoothness_check = true;
    float smoothness_threshold = 0.1;
    if (use_smoothness_check)
    {
        float expected_distance = ncross.norm () * seed_resolution; // 期望的距离 ： 叉积向量 长度 × 种子点精度
        float dot_p_1 = vec_t_to_s.dot (source_normal);
        float dot_p_2 = vec_s_to_t.dot (target_normal);
        float point_dist = (std::fabs (dot_p_1) < std::fabs (dot_p_2)) ? std::fabs (dot_p_1) : std::fabs (dot_p_2); // 实际距离: 最小的一个为两个带有方向的点的距离
        const float dist_smoothing = smoothness_threshold * voxel_resolution;// 补偿距离

        if (point_dist > (expected_distance + dist_smoothing)) // 距离过大，不平滑!!!!!!!!!!!!!!!!!!!!!!!!!!!
        {
            is_smooth &= false; 
        }
    }

// 凸凹关系判断=======================================================
// 点云完成超体聚类之后，对于过分割的点云需要计算不同的块之间凹凸关系。
// 凹凸关系通过 CC（Extended Convexity Criterion） 和 SC （Sanity criterion）判据来进行判断。
// 其中 CC 利用相邻两片中心连线向量与法向量夹角来判断两片是凹是凸。
// 显然，如果图中a1>a2则为凹，反之则为凸。
//                   a2  |
//                  ------|------                         ____|——|
//           a1 |                 |                              |        |
//          ____|        凸     |               __|______| 凹 |
//                 |__________|              |______________|
// https://images2015.cnblogs.com/blog/710098/201512/710098-20151207170359699-415614858.jpg

// 考虑到测量噪声等因素，需要在实际使用过程中引入门限值（a1需要比a2大出一定量）来滤出较小的凹凸误判。
// 此外，为去除一些小噪声引起的误判，还需要引入“第三方验证”，如果某块和相邻两块都相交，则其凹凸关系必相同。


// 如果相邻两面中，有一个面是单独的，cc判据是无法将其分开的。
// 举个简单的例子，两本厚度不同的书并排放置，视觉算法应该将两本书分割开。如果是台阶，则视觉算法应该将台阶作为一个整体。
// 本质上就是因为厚度不同的书存在surface-singularities。为此需要引入SC判据，来对此进行区分。
// Sanity Criterion: Check if definition convexity/concavity makes sense for connection of given patches
// 如图所示，相邻两面是否真正联通，是否存在单独面，与θ角（中心点连线（x1-x2） 与两平面交线（n1-n2） 的 夹角）有关，θ角越大，则两面真的形成凸关系的可能性就越大。
// 据此，可以设计SC判据：
// https://images2015.cnblogs.com/blog/710098/201512/710098-20151207172059308-1314515634.jpg

    bool use_sanity_check = true;
    float intersection_angle =  pcl::getAngle3D (ncross, vec_t_to_s, true); // 连线 和 交线 夹角 
    float min_intersect_angle = (intersection_angle < 90.) ? intersection_angle : 180. - intersection_angle;// 0～90

    float intersect_thresh = 57. * 1. / (1. + exp (-0.3 * (normal_angle - 33.)));// 阈值
    if (min_intersect_angle < intersect_thresh && use_sanity_check)// 夹角过小为  单独面 而不是联通面(形成凸/凹空间体)
    {
        is_convex &= false;
    }

// Convexity Criterion: Check if connection of patches is convex. If this is the case the two SuperVoxels should be merged.
    if ((pcl::getAngle3D(vec_t_to_s, source_normal) - pcl::getAngle3D(vec_t_to_s, target_normal)) <= 0)// a1 < a2 为凸关系
    {
        is_convex &= true;  // connection convex
    }
    else
    {
        is_convex &= (normal_angle < concavity_tolerance_threshold);  // concave connections will be accepted  if difference of normals is small
    }
    return (is_convex && is_smooth);


// 在标记完各个小区域的凹凸关系后，则采用区域增长算法将小区域聚类成较大的物体。此区域增长算法受到小区域凹凸性限制，既：

// 只允许区域跨越凸边增长。
// 至此，分割完成，在滤去多余噪声后既获得点云分割结果。
// 此外：考虑到RGB-D图像随深度增加而离散，难以确定八叉树尺寸，故在z方向使用对数变换以提高精度。
// 分割结果如图： https://images2015.cnblogs.com/blog/710098/201512/710098-20151207183006543-245415125.jpg
}


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
