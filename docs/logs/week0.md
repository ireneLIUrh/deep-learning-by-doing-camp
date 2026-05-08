# Week 0：项目启动

## 本周目标

- [x] 创建 GitHub 仓库
- [x] 在实验室服务器上 clone 仓库
- [x] 创建新的 conda 环境
- [x] 安装 MkDocs Material
- [x] 搭建第一版文档站
- [x] 发布到 GitHub Pages
- [x] 开启线性代数复习
- [x] 完成第一次正式 commit

## 本项目为什么存在？

我希望通过 learning by doing 的方式学习深度学习。

这个项目不仅用于个人学习，也希望未来可以整理成适合同样想入门深度学习的人使用的课程式材料。

## Week 0 的完成标准

本周结束时，应当能够打开一个 GitHub Pages 网站，并看到：

- 首页
- Week 0 学习日志
- 线性代数复习页面
- 深度学习路线图
- 代码实践说明

## Week 0 复盘

### 已完成

- 创建并配置 GitHub 仓库
- 在服务器上完成项目 clone
- 使用 SSH 解决 GitHub push 问题
- 搭建 MkDocs Material 文档站
- 完成本地预览和 GitHub Pages 部署

### 遇到的问题

- conda 解析依赖较慢
- pip 安装较新 pandas 时触发源码编译，老服务器 GCC 不兼容
- Git 1.8.3.1 的 `git add .` 行为较旧
- HTTPS push GitHub 失败，最终改用 SSH

### 当前经验

- 老服务器上优先使用 SSH 推送 GitHub
- `git add -A` 比 `git add .` 更稳妥
- requirements 应固定版本，避免 pip 自动安装过新的科学计算包
- Markdown 编辑应优先使用 VS Code / github.dev，而不是 nano 或 cat