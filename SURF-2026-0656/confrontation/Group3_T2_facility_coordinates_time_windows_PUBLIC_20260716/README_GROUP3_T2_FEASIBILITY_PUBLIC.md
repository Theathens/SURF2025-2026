# 给 Group3/T2 的厂区经纬度与时间段可行性检查包

生成日期：2026-07-16

## 这份文件的用途

这份材料只用于请 Group3 的 T2 判断：这些厂区点位能不能做遥感 AOI 确认和时间序列特征提取。文件内容限定为公开定位信息、建议时间窗和可行性问题。

## 文件

- `group3_t2_core_facility_coordinates_time_windows_PUBLIC.csv`
  - 当前 Round2 主样本厂区，共 9 个。
  - 可直接优先给 T2 看。

- `group3_t2_optional_facility_candidates_for_feasibility_PUBLIC.csv`
  - 额外候选点，共 12 个。
  - 只用于问 T2 是否值得继续做，不代表已经进入主数据。

## 需要 T2 帮忙判断的问题

1. 经纬度是否足够让 T2 找到厂区 AOI？
2. 如果不能直接画 AOI，需要 Group1/T1 再补什么：更精确地址、园区名称、厂界、多边形，还是高清地图截图？
3. 建议用哪些数据源：Sentinel-2、Landsat、VIIRS 夜光、MODIS/Landsat LST？
4. 建议的 baseline / detection 时间段是否合理？
5. 哪些厂区最适合作为 Round2 遥感交叉验证样本，哪些应该暂缓？

## 重要说明

- 所有坐标都是 WGS-84。
- 目前大多数点是 `point_seed_only`，不是精确厂界。
- 遥感 proxy 还没有处理；表里没有 built-up area、roof PV、night light、NDVI、heat index 的测量值。
- 如果 T2 认为某些点位无法稳定定位，Group1 会继续补更精确地址或改用其他厂区。
