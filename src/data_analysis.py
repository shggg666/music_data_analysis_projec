import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import os

# 创建结果目录
os.makedirs('../results/analysis_plots', exist_ok=True)
os.makedirs('../results/model_metrics', exist_ok=True)

# 1. 数据加载与预处理
def load_data(file_path):
    """加载音乐数据集并进行基础预处理"""
    try:
        data = pd.read_csv(file_path)
        print(f"数据加载完成，共{len(data)}条记录")
        print(f"数据集列名: {list(data.columns)}")
        
        # 检查缺失值
        missing_values = data.isnull().sum()
        if missing_values.sum() > 0:
            print("\n缺失值统计:")
            print(missing_values[missing_values > 0])
            
            # 简单填充缺失值
            data = data.fillna(data.mean())
            print("\n已使用均值填充缺失值")
        
        return data
    except Exception as e:
        print(f"数据加载失败: {e}")
        return pd.DataFrame()

# 2. 探索性数据分析
def exploratory_analysis(data):
    """执行探索性数据分析并保存结果图表"""
    if data.empty:
        return
    
    # 2.1 流派分布
    genre_counts = data['genre'].value_counts()
    print("\n各流派歌曲数量:")
    print(genre_counts)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=genre_counts.values, y=genre_counts.index)
    plt.title('各流派歌曲数量分布')
    plt.xlabel('歌曲数量')
    plt.tight_layout()
    plt.savefig('../results/analysis_plots/genre_distribution.png')
    plt.close()
    
    # 2.2 特征相关性分析
    numeric_features = data.select_dtypes(include=['float64', 'int64'])
    corr_matrix = numeric_features.corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('特征相关性热图')
    plt.tight_layout()
    plt.savefig('../results/analysis_plots/correlation_heatmap.png')
    plt.close()
    
    # 2.3 不同流派的特征对比
    features_to_compare = ['danceability', 'energy', 'valence', 'loudness']
    for feature in features_to_compare:
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='genre', y=feature, data=data)
        plt.title(f'不同流派的{feature}特征对比')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'../results/analysis_plots/{feature}_comparison.png')
        plt.close()
    
    return genre_counts

# 3. 音乐流派分类模型
def train_genre_classifier(data):
    """训练音乐流派分类模型并评估性能"""
    if data.empty:
        return None, None, None
    
    # 准备特征和目标变量
    numeric_features = data.select_dtypes(include=['float64', 'int64'])
    X = numeric_features.drop(['genre'], axis=1) if 'genre' in numeric_features.columns else numeric_features
    y = data['genre'] if 'genre' in data.columns else pd.Series()
    
    if X.empty or y.empty:
        print("无法构建特征矩阵或目标变量")
        return None, None, None
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 训练随机森林分类器
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 模型预测
    y_pred = model.predict(X_test)
    
    # 模型评估
    report = classification_report(y_test, y_pred)
    print("\n模型评估报告:")
    print(report)
    
    # 保存评估报告
    with open('../results/model_metrics/classification_report.txt', 'w') as f:
        f.write(report)
    
    # 特征重要性
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
    plt.title('特征重要性排名')
    plt.tight_layout()
    plt.savefig('../results/analysis_plots/feature_importance.png')
    plt.close()
    
    return model, X_test, y_test

# 4. 主函数
if __name__ == "__main__":
    print("===== 音乐数据分析项目 =====")
    
    # 加载数据
    data = load_data("../data/music_data.csv")
    
    # 探索性分析
    genre_counts = exploratory_analysis(data)
    
    # 训练分类模型
    model, X_test, y_test = train_genre_classifier(data)
    
    print("\n分析完成！结果已保存至results/目录")    
