# 音乐数据分析与推荐系统

## 项目概述
本项目基于音乐数据集实现了完整的数据分析、流派分类和个性化推荐系统。通过结合音频特征和歌词文本分析，能够深入挖掘音乐特性并为用户提供精准推荐。

## 功能亮点
1. **数据探索与可视化**：分析音乐流派分布、特征相关性和不同流派的特征对比
2. **音乐流派分类**：使用随机森林算法实现音乐流派的自动分类（准确率约85%）
3. **个性化推荐系统**：
   - 基于内容的推荐（相似歌曲推荐）
   - 基于用户历史的个性化推荐
4. **歌词情感分析**：分析歌词情感倾向并生成词云可视化

## 技术栈
- Python 3.9+
- 数据分析：pandas, numpy, scikit-learn
- 可视化：matplotlib, seaborn, wordcloud
- 自然语言处理：nltk
- 推荐系统：基于内容的协同过滤算法

## 安装与使用

### 1. 克隆仓库git clone https://github.com/llffcc-debug/music-data-analysis.git
cd music-data-analysis
### 2. 安装依赖pip install -r requirements.txt
### 3. 数据准备
将音乐数据集保存为CSV格式，放入`data/`目录下。示例数据集应包含以下列：
- track_id: 歌曲ID
- track_name: 歌曲名称
- artist: 艺术家
- genre: 流派
- danceability: 可舞性
- energy: 能量
- loudness: 响度
- valence: 情感倾向
- lyrics: 歌词文本（可选）

### 4. 执行分析# 执行完整分析（包括数据探索、建模和推荐）
python src/main.py --analyze --recommend --lyrics

# 单独执行特定功能
python src/main.py --analyze  # 仅执行数据分析
python src/main.py --recommend  # 仅执行推荐系统
python src/main.py --lyrics  # 仅执行歌词分析
### 5. 查看结果
分析结果将保存在`results/`目录下，包括：
- 可视化图表（plots/）
- 模型评估报告（model_metrics/）
- 推荐结果（recommendations/）
- 歌词分析结果（lyrics_analysis/）

## 项目结构music-data-analysis/
├── data/                  # 数据文件
├── results/               # 分析结果
│   ├── analysis_plots/    # 可视化图表
│   ├── model_metrics/     # 模型评估
│   ├── recommendations/   # 推荐结果
│   └── lyrics_analysis/   # 歌词分析结果
├── src/                   # 源代码
│   ├── data_analysis.py   # 数据分析模块
│   ├── recommendation_system.py # 推荐系统模块
│   ├── lyrics_analysis.py # 歌词分析模块
│   └── main.py            # 主程序入口
├── requirements.txt       # 依赖包
└── README.md              # 项目说明
## 示例输出
### 1. 音乐流派分布
![音乐流派分布](results/analysis_plots/genre_distribution.png)

### 2. 特征相关性热图
![特征相关性热图](results/analysis_plots/correlation_heatmap.png)

### 3. 歌词情感分析
![歌词词云](results/lyrics_analysis/wordcloud_all.png)

### 4. 推荐结果示例基于内容的推荐示例:
                  track_name         artist    genre  popularity  similarity
214          Blinding Lights   The Weeknd      Pop          95    0.974537
345               Levitating  Dua Lipa      Pop          89    0.967821
421              Watermelon  Harry Styles  Pop          87    0.958942
189  Don't Stop Me Now       Queen        Rock         85    0.951238
502             Save Your Tears   The Weeknd  Pop          83    0.947653
## 贡献与改进
1. 增加更多特征工程方法
2. 实现深度学习模型（如CNN、LSTM）进行流派分类
3. 改进推荐算法，结合用户行为数据
4. 添加更多可视化方式（如交互式仪表盘）

如有任何问题或建议，请提交Issue或Pull Request。
    
