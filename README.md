
# orb-slam2_with_semantic_label  yolo-v2 语义-slam

# 结合要点!!!!!!
 
    整合orbslam2-pc 和 yolov3-se
[orbslam2-pc](https://github.com/Ewenwan/ORBSLAM2_with_pointcloud_map)

[yolov3-se](https://github.com/Ewenwan/YOLOv3_SpringEdition)

# 相比 orbslam2-pc 变动的地方

## 1. 头文件部分 
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
        color_importance (1.0f),      // 颜色重要性权重
        spatial_importance (0.4f),    // 空间重要性权重 
        normal_importance  (1.0f),    // 法线重要性权重
        use_single_cam_transform (false),
        use_supervoxel_refinement (false),

        // Default parameters for model fitting
        use_random_sampling (false),
        noise_threshold(0.02f),
        smooth_cost (0.001),
        min_inliers_per_plane (100),
       	min_plane_area(0.025),
        max_num_iterations (25),
        max_curvature (0.01f),        // 是否为平面的 曲率 阈值
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

        voxel_resolution=config.voxel_resolution; // 体素格子大小
        seed_resolution=config.seed_resolution;   // 种子大小  超体聚类算法参数
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
  
  std::vector<cv::Scalar> colors;// 图像颜色，图像个通道值。。。。用处大大滴。。。目标检测不同类别显示不同颜色
  
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
//            a2    |
//            ------|------                    ____|—— |
//           a1 |         |                        |   |
//          ____|   凸    |               __|______|凹 |
//              |_________|              |_____________|
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

## 2.源文件部分
> pointcloudmapping.cc 修改
```c
// 增添的一大堆都文件，和 pointcloudmapping.h 有部分重复，都可以放在，pointcloudmapping.h 头文件处

// yolov3 目标检测需要使用 GPU加速
#ifdef GPU
#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"
#endif



// 主要修改函数 ======
void PointCloudMapping::viewer()
{
// 前面等待 关键帧更新 线程

// 后面遍例每一个关键帧先执行 目标检测
// 在彩色图上 按照检测结果画填充 不同颜色(和类别物体 有一一对应关系)的 矩形
// 按照 带有分类矩形框的 彩色图 和 深度图 产生 xyzrgba 点云，
// 后面的到 总的点云后根据 rgb颜色来区分不同种类物体， 形成 xyzl 带有标签的点云
}

```

> segmentation.cc 点云分割算法
```c

void Segmentation::doSegmentation()
{

  // Start the clock
  clock_t sv_start = clock();

  // Preparation of Input: Supervoxel Oversegmentation

/////////////////////////////////////////////////////////////////////////////////////////////////////////////
// 步骤1,自底向上，超体聚类，过分割=====================================
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
  pcl::SupervoxelClustering<PointT> super (config_.voxel_resolution, config_.seed_resolution);
  super.setUseSingleCameraTransform (config_.use_single_cam_transform);
  super.setInputCloud (input_cloud_ptr_); // 输入点云
  super.setColorImportance (config_.color_importance);         // 颜色重要性权重
  super.setSpatialImportance (config_.spatial_importance);   // 空间重要性权重 
  super.setNormalImportance (config_.normal_importance);// 法线重要性权重
  std::map<uint32_t, pcl::Supervoxel<PointT>::Ptr> supervoxel_clusters;// id:超体点云指针 对

  super.extract (supervoxel_clusters); // 执行超体聚类

  if (config_.use_supervoxel_refinement)
  {
    PCL_INFO ("Refining supervoxels\n");
    super.refineSupervoxels (2, supervoxel_clusters);// 微调
  }
  std::cout << "Number of supervoxels: " << supervoxel_clusters.size () << "\n";// 超体聚类后的到的 超体数量

  std::multimap<uint32_t, uint32_t> supervoxel_adjacency; // 超体链接关系 多对多 multimap
  super.getSupervoxelAdjacency (supervoxel_adjacency);    // 计算超体链接关系 
   // 计算超体中心 的法线 
  pcl::PointCloud<pcl::PointNormal>::Ptr sv_centroid_normal_cloud = pcl::SupervoxelClustering<PointT>::makeSupervoxelNormalCloud (supervoxel_clusters);
  clock_t sv_end = clock();
  printf("Super-voxel segmentation takes: %.2fms\n", (double)(sv_end - sv_start)/(CLOCKS_PER_SEC/1000));

/////////////////////////////////////////////////////////////////////////////////////////////////////////////
// 步骤2 Constrained plane extraction   分割平面======================================
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /* Generate plane hypotheses using super-voxels  生成平面假设*/
  std::vector<Eigen::Vector4f> planes_coeffs;
  std::vector<Eigen::Vector3f> planes_hough;
  double min_theta = 360; double max_theta = -360;
  double min_phi = 360; double max_phi = -360;
  double min_rho = 100; double max_rho = -100;

  std::map<uint32_t, pcl::Supervoxel<PointT>::Ptr>::iterator cluster_itr_c = supervoxel_clusters.begin();
  for (; cluster_itr_c != supervoxel_clusters.end(); cluster_itr_c++)
  { // 103
    pcl::Supervoxel<PointT>::Ptr sv = cluster_itr_c->second;// 每一个超体
    pcl::PointCloud<PointT>::Ptr cloud = sv->voxels_;            // 对应的点云团
    float curvature; // 面曲率 法线估计输出 
    Eigen::Vector4f plane_par;  // 平面参数 法线估计输出  nx,ny,nz,nc, 前三个表示法线向量
    Eigen::Vector3f hough_par;// 
    pcl::computePointNormal(*cloud, plane_par, curvature);
    if (curvature < config_.max_curvature) // 曲率较小的为 平面
    {
      // Convert to Hough transform  霍夫变换
      double theta = std::atan(plane_par(1)/plane_par(0))*180/M_PI; // xoy平面投影 与y轴夹角 
      double phi = std::acos(plane_par(2))*180/M_PI;// 法线模长为1, 法线与z轴夹角
      double rho = plane_par(3);
      if (isnan(theta) | isnan(phi) | isnan(rho)) continue;
      hough_par(0) = theta;
      hough_par(1) = phi;
      hough_par(2) = rho;
      if (theta < min_theta) min_theta = theta;
      if (theta > max_theta) max_theta = theta;
      if (phi < min_phi) min_phi = phi;
      if (phi > max_phi) max_phi = phi;
      if (rho < min_rho) min_rho = rho;
      if (rho > max_rho) max_rho = rho;
      planes_hough.push_back(hough_par);
      planes_coeffs.push_back(plane_par);
    }
  }

  // Plane hypothesis generation using random sampling  随机采样一致性 生成平面假设
  if (config_.use_random_sampling)
  { // 149 
    std::cout << "Randomly sampling plane hypotheses...\n";
    int max_random_hyps = 1000; //
    int count = 0;
    int num_supervoxels = supervoxel_clusters.size ();
    srand(time(NULL));
    pcl::SampleConsensusModelPlane<pcl::PointNormal>::Ptr model_p (new pcl::SampleConsensusModelPlane<pcl::PointNormal> (sv_centroid_normal_cloud));
    while (count < max_random_hyps)
    {
      // random sample 4 points
      std::vector<int> samples;
      int iters;
      model_p->getSamples(iters, samples); // 随机采样部分点
      Eigen::VectorXf plane_par;
      model_p->computeModelCoefficients(samples, plane_par);// 计算模型参数

      std::set<int> test_points;// 生成测试点集
      test_points.insert((int)(rand()%num_supervoxels+1));
      bool good_model = model_p->doSamplesVerifyModel(test_points, plane_par, config_.noise_threshold*0.5);// 验证模型
      if (good_model == false) continue;// 该采样的到的模型不好，跳过

      Eigen::Vector3f hough_par;
      double theta = std::atan(plane_par(1)/plane_par(0))*180/M_PI;
      double phi = std::acos(plane_par(2))*180/M_PI;
      double rho = plane_par(3);

      if (isnan(theta) | isnan(phi) | isnan(rho)) continue;

      planes_coeffs.push_back(plane_par);
      hough_par(0) = theta;
      hough_par(1) = phi;
      hough_par(2) = rho;

      if (theta < min_theta) min_theta = theta;
      if (theta > max_theta) max_theta = theta;
      if (phi < min_phi) min_phi = phi;
      if (phi > max_phi) max_phi = phi;
      if (rho < min_rho) min_rho = rho;
      if (rho > max_rho) max_rho = rho;
      planes_hough.push_back(hough_par);
      count++;
    }
  }

  // Assign points to planes.
  uint32_t node_ID = 0;
  std::map<uint32_t, uint32_t> label2index;
  std::map<uint32_t, pcl::Supervoxel<PointT>::Ptr>::iterator cluster_itr = supervoxel_clusters.begin();
  for (; cluster_itr != supervoxel_clusters.end(); cluster_itr++)
  {
    label2index[cluster_itr->first] = node_ID;
    node_ID++;
  }
  uint32_t num_super_voxels = sv_centroid_normal_cloud->size();
  std::vector<int> sv_labels(num_super_voxels,0);
  int good_planes_count = 0;
  uint32_t outlier_label = 0;
  // Remove duplicated planes
  if (planes_coeffs.size() > 0)
  { // 384 结束
    std::vector<Eigen::Vector4f> plane_candidates;
    double step_theta = 1;
    double step_phi = 1;
    double step_rho = 0.025;
    int theta_bins = round((max_theta - min_theta)/step_theta) + 1;
    int phi_bins = round((max_phi - min_phi)/step_phi) + 1;
    int rho_bins = round((max_rho - min_rho)/step_rho) + 1;

    unsigned char*** accumulator;
    accumulator = new unsigned char**[theta_bins];
    if (accumulator != NULL){
      for (int i=0; i<theta_bins; i++){
        accumulator[i] = new unsigned char*[phi_bins];
        if (accumulator[i] != NULL)
        for (int j=0; j<phi_bins; j++)
        accumulator[i][j] = new unsigned char[rho_bins];
      }
    }

    for (int i=0; i<planes_coeffs.size(); i++){
      Eigen::Vector3f hough_par = planes_hough.at(i);
      int b_theta = floor((hough_par(0) - min_theta + 0.0001)/step_theta);
      int b_phi = floor((hough_par(1) - min_phi + 0.0001)/step_phi);
      int b_rho = floor((hough_par(2) - min_rho + 0.0001)/step_rho);

      if (accumulator[b_theta][b_phi][b_rho] != 1){
        accumulator[b_theta][b_phi][b_rho] = 1;
        plane_candidates.push_back(planes_coeffs.at(i));
      }
    }
    // Free accumulator memory
    for (int i = 0; i < theta_bins; i++){
      for (int j = 0; j < phi_bins; j++){
        delete [] accumulator[i][j];
      }
      delete [] accumulator[i];
    }
    delete [] accumulator;
    std::cout << "Number of planes remained after hough-based filtering = " << plane_candidates.size() << "\n";
    // Compute plane unary costs
    float min_num_supervoxel_per_plane = config_.min_plane_area/(config_.seed_resolution*config_.seed_resolution/4/M_PI);
    std::vector<Eigen::Vector4f> good_planes;
    std::vector<Eigen::VectorXi> planes_inliers_idx;
    std::vector<float> unaries;
    uint32_t num_planes = plane_candidates.size();
    Eigen::MatrixXi inliers_mat(num_planes, num_super_voxels);
    Eigen::MatrixXf normals_mat(num_planes, num_super_voxels);
    Eigen::MatrixXf point2plane_mat(num_planes, num_super_voxels);
    int count_idx = 0;
    for (int j = 0; j<num_planes; ++j){
      Eigen::Vector4f p_coeffs = plane_candidates.at(j);

      Eigen::Vector3f p_normal;
      p_normal[0] = p_coeffs[0];
      p_normal[1] = p_coeffs[1];
      p_normal[2] = p_coeffs[2];
      Eigen::VectorXi inliers_idx(num_super_voxels);
      Eigen::VectorXf point2plane(num_super_voxels);
      inliers_idx = Eigen::VectorXi::Zero(num_super_voxels);
      int inliers_count = 0;
      float plane_score = 0;
      for (size_t i = 0; i < num_super_voxels; ++i){
        pcl::PointXYZ p;
        Eigen::Vector3f n;
        p.x = sv_centroid_normal_cloud->at(i).x;
        p.y = sv_centroid_normal_cloud->at(i).y;
        p.z = sv_centroid_normal_cloud->at(i).z;
        n[0] = sv_centroid_normal_cloud->at(i).normal_x;
        n[1] = sv_centroid_normal_cloud->at(i).normal_y;
        n[2] = sv_centroid_normal_cloud->at(i).normal_z;

        // Distance from a point to a plane is scaled with a weight measuring the difference between point and plane normals.
        float p2p_dis = pcl::pointToPlaneDistance(p,p_coeffs);
        if (std::isnan(p2p_dis)) p2p_dis = config_.noise_threshold;
        float dotprod = std::fabs(n.dot(p_normal));
        if (std::isnan(dotprod)) dotprod = 0;
        float normal_dis = dotprod < 0.8 ? 100 : 1;
        float data_cost = p2p_dis*normal_dis;
        plane_score += -std::exp(-data_cost/(2*config_.noise_threshold)); // -1 is best, 0 is worst
        point2plane(i) = data_cost;
        if (data_cost <= config_.noise_threshold) {
          inliers_idx(i) = 1;
          inliers_count++;
        }
      }
      plane_score = plane_score/(min_num_supervoxel_per_plane*10);
      float confidence_threshold = -0.1f;
      if (plane_score <= confidence_threshold){
        inliers_mat.row(count_idx) = inliers_idx;
        normals_mat.row(count_idx) << p_coeffs(0), p_coeffs(1), p_coeffs(2);
        point2plane_mat.row(count_idx) = point2plane;
        good_planes.push_back(p_coeffs);
        planes_inliers_idx.push_back(inliers_idx);
        double u_cost = plane_score;
        unaries.push_back(u_cost);
        count_idx++;
      }
    }
    clock_t plane_sampling_end = clock();
    printf("Plane generation takes: %.2fms\n", (double)(plane_sampling_end - sv_end)/(CLOCKS_PER_SEC/1000));
    num_planes = unaries.size();
    std::cout << "Number of plane candidates = " << num_planes << "\n";
    if (num_planes > 1)
   {

      inliers_mat.conservativeResize(num_planes, num_super_voxels);
      normals_mat.conservativeResize(num_planes, num_super_voxels);
      point2plane_mat.conservativeResize(num_planes, num_super_voxels);

      Eigen::VectorXi inliers_count = inliers_mat.rowwise().sum();
      //Eigen::MatrixXi ov_mat = inliers_mat*inliers_mat.transpose();
      //Eigen::MatrixXf dot_mat = normals_mat*normals_mat.transpose();

      Eigen::MatrixXi temp1 = inliers_mat.transpose();
      Eigen::MatrixXi ov_mat = inliers_mat*temp1;
      Eigen::MatrixXf temp2 = normals_mat.transpose();
      Eigen::MatrixXf dot_mat = normals_mat*temp2;

      Eigen::VectorXf plane_unaries = Eigen::Map<Eigen::MatrixXf>(unaries.data(), num_planes, 1);
      Eigen::MatrixXf plane_pairwises = Eigen::MatrixXf::Zero(num_planes, num_planes);

      for (int i=0; i<num_planes-1; i++){
        for (int j=i+1; j<num_planes; j++){
          double ov_cost = (double)ov_mat(i,j)/std::min(inliers_count(i), inliers_count(j));
          //if (ov_cost > 0.75) ov_cost = 1.0;
          //else ov_cost = 0;
          double dot_prod = std::abs(dot_mat(i,j));
          dot_prod = dot_prod < 0.5 ? dot_prod : 1 - dot_prod;
          double angle_cost = 1 - exp(-dot_prod/0.25);
          double p_cost;
          //if (ov_cost == 0) p_cost = 0; // TODO: If the two planes do not intersect, we do not constraint them!
          //else p_cost = angle_cost; ;//0.5*angle_cost + 0.5*ov_cost;
          p_cost = angle_cost;
          plane_pairwises(i,j) = p_cost*0.5;
          plane_pairwises(j,i) = p_cost*0.5;
        }
      }

      Eigen::VectorXi initLabeling(num_planes);
      initLabeling = Eigen::VectorXi::Ones(num_planes);
      Eigen::VectorXi finalLabeling(num_planes);
      double finalEnergy = 0;
     // LSA_tr 算法 平面分割==============================================================
      LSA_TR(&finalEnergy, &finalLabeling, num_planes, plane_unaries, plane_pairwises, initLabeling);
      if (finalEnergy == 0)
      {
        PCL_WARN("Optimization got stuck \n");
        finalLabeling = Eigen::VectorXi::Ones(num_planes);
      }
      int num_selected_planes = finalLabeling.sum();
      std::cout << "Number of supporting planes detected = " << num_selected_planes << "\t";
      std::cout << "(Note: This is not the true number of planes in the scene.)\n";
      std::vector<Eigen::Vector4f> selected_planes;
      Eigen::MatrixXf unary_matrix(num_selected_planes + 1, num_super_voxels);
      int sidx = 1;
      for (int i=0; i<num_planes; i++){
        if (finalLabeling(i) == 1) {
          selected_planes.push_back(good_planes.at(i));
          unary_matrix.row(sidx) = point2plane_mat.row(i)*config_.gc_scale;
          sidx++;
        }
      }

      // Outlier data cost  外点 数据损失 =====================
      int num_labels = num_selected_planes + 1;
      outlier_label = 0;
      unary_matrix.row(outlier_label) = Eigen::VectorXf::Ones(num_super_voxels)*((config_.noise_threshold*config_.gc_scale));
      Eigen::MatrixXi unary_matrix_int = unary_matrix.cast<int>();
      int *data_cost = new int[num_super_voxels*num_labels];
      Eigen::Map<Eigen::MatrixXi>(data_cost, unary_matrix_int.rows(), unary_matrix_int.cols() ) = unary_matrix_int;
     //  图割算法计算损失==================================================================
      GCoptimizationGeneralGraph *gc = new GCoptimizationGeneralGraph(num_super_voxels, num_labels);
      gc->setDataCost(data_cost);

      for ( int l1 = 0; l1 < num_labels; l1++){
        for (int l2 = 0; l2 < num_labels; l2++){
          if (l1==0) gc->setSmoothCost(l1,l2,0);
          if (l1==l2) gc->setSmoothCost(l1,l2,0);
          else {
            gc->setSmoothCost(l1,l2,1);
          }
        }
      }

      std::multimap<uint32_t,uint32_t>::iterator adjacency_itr = supervoxel_adjacency.begin();
      float smooth_cost = config_.noise_threshold/2;
      for ( ; adjacency_itr != supervoxel_adjacency.end(); ++adjacency_itr)
      {
        uint32_t node1 = label2index[adjacency_itr->first];
        uint32_t node2 = label2index[adjacency_itr->second];
        Eigen::Vector3f n1;
        n1[0] = sv_centroid_normal_cloud->at(node1).normal_x;
        n1[1] = sv_centroid_normal_cloud->at(node1).normal_y;
        n1[2] = sv_centroid_normal_cloud->at(node1).normal_z;
        Eigen::Vector3f n2;
        n2[0] = sv_centroid_normal_cloud->at(node2).normal_x;
        n2[1] = sv_centroid_normal_cloud->at(node2).normal_y;
        n2[2] = sv_centroid_normal_cloud->at(node2).normal_z;
        float w = std::fabs(n1.dot(n2));
        if (w < 0.5) w = 0.0f;
        int edge_weight = (int)(w * config_.gc_scale * smooth_cost); // This works better than Potts smoothness model
        gc->setNeighbors(node1,node2,edge_weight);
      }
      try{
        gc->expansion(config_.max_num_iterations);
      }catch(GCException e){
        e.Report();
      }

      for (int i=0; i<num_super_voxels;i++){
        sv_labels.at(i) = gc->whatLabel(i);
      }
      // Free some memory
      delete gc;
      delete data_cost;
      //////////////////////////////////////////////////////////////////////////////////////////////////////////////
      // End of constrained plane extraction
      //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    }
    clock_t plane_fitting_end = clock();
    printf("Global plane extraction takes: %.2fms\n", (double)(plane_fitting_end - plane_sampling_end)/(CLOCKS_PER_SEC/1000));
  }

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// 步骤3： Segment the point cloud into objects
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
  std::cout << "Run graph based object segmentation using the extracted planes\n";
  using namespace boost;
  std::vector<uint32_t> supervoxel_labels;
  { // 460 结束
    typedef adjacency_list <vecS, vecS, undirectedS> Graph;

    Graph G;
    for (uint32_t i = 0; i < supervoxel_clusters.size(); ++i)
    {
      add_vertex(G);
    }

    std::multimap<uint32_t,uint32_t>::iterator adjacency_itr = supervoxel_adjacency.begin();
    for ( ; adjacency_itr != supervoxel_adjacency.end(); ++adjacency_itr)
    {
      uint32_t from = label2index[adjacency_itr->first];
      uint32_t to = label2index[adjacency_itr->second];

      uint32_t label_from = sv_labels.at(from);
      uint32_t label_to = sv_labels.at(to);

      Eigen::Vector3f p1;
      p1[0] = sv_centroid_normal_cloud->at(from).x;
      p1[1] = sv_centroid_normal_cloud->at(from).y;
      p1[2] = sv_centroid_normal_cloud->at(from).z;

      Eigen::Vector3f n1;
      n1[0] = sv_centroid_normal_cloud->at(from).normal_x;
      n1[1] = sv_centroid_normal_cloud->at(from).normal_y;
      n1[2] = sv_centroid_normal_cloud->at(from).normal_z;

      Eigen::Vector3f p2;
      p2[0] = sv_centroid_normal_cloud->at(to).x;
      p2[1] = sv_centroid_normal_cloud->at(to).y;
      p2[2] = sv_centroid_normal_cloud->at(to).z;

      Eigen::Vector3f n2;
      n2[0] = sv_centroid_normal_cloud->at(to).normal_x;
      n2[1] = sv_centroid_normal_cloud->at(to).normal_y;
      n2[2] = sv_centroid_normal_cloud->at(to).normal_z;

      if (label_from != label_to) continue;
      if (label_from == label_to && label_from != outlier_label){
        add_edge(from,to,G);
        continue;
      }
      /////////  凸凹性判断=============================================================
      bool convex = isConvex(p1, n1, p2, n2, config_.seed_resolution, config_.voxel_resolution);
      if (convex == true) add_edge(from,to,G);
      //sv_centroid_normal_cloud
    }

    std::vector<uint32_t> component(num_vertices(G));

    // 链接组建========================================
    uint32_t num = connected_components(G, &component[0]);
    std::cout << "Number of connected components: " << num <<"\n";

    int min_voxels_per_cluster = 2;
    int outlier_label = 0;
    std::map<uint32_t,uint32_t> label_list_map;
    int new_label = 1;
    for (uint32_t i = 0; i != component.size(); ++i){
      int count = std::count(component.begin(), component.end(), component[i]);
      int label = component[i];
      if (label_list_map.find(label) == label_list_map.end() && count >= min_voxels_per_cluster){ // label not found
        label_list_map[label] = new_label;
        new_label++;
      }
      if (count < min_voxels_per_cluster){ // minimum number of supervoxels in each component
        supervoxel_labels.push_back(outlier_label);
      }
      else
      supervoxel_labels.push_back(label_list_map.find(label)->second);
    }
  }
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // End of scene segmentation
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  // Re-label the point cloud
  segmented_cloud_ptr_ = super.getLabeledCloud ();
  std::map<uint32_t,uint32_t> label_to_seg_map;
  std::map<uint32_t, pcl::Supervoxel<PointT>::Ptr>::iterator cluster_itr_ = supervoxel_clusters.begin();
  uint32_t idx = 0;
  for (; cluster_itr_ != supervoxel_clusters.end(); cluster_itr_++)
   {
    //label_to_seg_map[cluster_itr_->first] = sv_labels.at(idx); // Use this to plot plane segmentation only
    label_to_seg_map[cluster_itr_->first] = supervoxel_labels.at(idx);
    idx++;
  }
  typename pcl::PointCloud<pcl::PointXYZL>::iterator point_itr = (*segmented_cloud_ptr_).begin();
  uint32_t zero_label = 0;
    uint32_t count = 0;
  for (; point_itr != (*segmented_cloud_ptr_).end(); ++point_itr)
  {
    if (point_itr->label == 0)
    {
      zero_label++;
      count++;

    }
    else
    {
      point_itr->label = label_to_seg_map[point_itr->label];
      count++;
    }
  }
  cout <<count<<endl;


 printf("All Time taken: %.2fms\n", (double)(clock() - sv_start)/(CLOCKS_PER_SEC/1000));
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
