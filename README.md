# 战争进展采集分析平台 (War Watch Platform)

## 最新更新 (v2.1)
- **启动优化**：修复了桌面快捷方式可能闪退的问题。现在的启动窗口会保持开启，并显示清晰的执行状态。
- **数据源增强**：
  - 新增 **Fox News** 等右翼/保守派媒体，平衡政治光谱。
  - 媒体源现在带有 **政治光谱 (Political Spectrum)**（如中立、左翼、保守派、官方喉舌）和 **独立性 (Independence)**（如公共资金、企业所有、国家控制）标签。
- **时间轴升级**：
  - 采用 **Altair 交互式图表**，支持鼠标悬停查看详情（包含当地时间、事件类型、来源）。
  - 新增 **领导人表态** 专属事件类型，自动识别普京、泽连斯基、拜登、特朗普、内塔尼亚胡等关键人物的发言。
- **领导人追踪**：
  - 关键词库已覆盖美、俄、乌、伊、以、英、法、德等国主要领导人。

## 快速启动
1. **桌面快捷方式**：
   - 请查看您的桌面，已为您创建名为 `War Watch Platform` 的图标。
   - 双击即可启动平台。
   - *注：如果之前双击闪退，请重试，新脚本已修复此问题。*
   
2. **手动启动**：
   - 进入 `war_watch_platform` 目录。
   - 右键点击 `start_desktop.ps1` -> "使用 PowerShell 运行"。

## 功能模块
- **每日双语总结**：自动聚合各方视角，生成中英对照摘要。
- **交互式时间轴**：可视化展示冲突升级/降级趋势，高亮重大事件。
- **深度分析列表**：包含详细的来源链接、智能影响分析及媒体背景信息。

## 数据源列表 (部分)
- **中国**: 新华社, Global Times, CGTN
- **美国**: CNN (Left-lean), Fox News (Right-lean), AP (Center), VOA (Gov)
- **俄罗斯**: TASS, RT, MID
- **乌克兰**: Ukrinform, Kyiv Independent, MoD
- **伊朗**: Press TV, Tasnim, Mehr
- **以色列**: Times of Israel, Jerusalem Post, Haaretz (Left-lean)
- **欧洲**: BBC, Guardian, Sky News, France 24, RFI, Le Monde, DW, Spiegel
