import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

# 创建结果目录
os.makedirs('../results/recommendations', exist_ok=True)

class MusicRecommendationSystem:
    def __init__(self, data):
        """初始化音乐推荐系统"""
        self.data = data
        self.similarity_matrix = None
        self.track_indices = None
        
    def prepare_features(self):
        """准备用于计算相似度的特征"""
        if self.data.empty:
            return pd.DataFrame()
        
        # 选择用于推荐的特征
        feature_columns = [
            'danceability', 'energy', 'key', 'loudness', 'mode', 
            'speechiness', 'acousticness', 'instrumentalness', 
            'liveness', 'valence', 'tempo'
        ]
        
        # 检查特征列是否存在
        available_features = [col for col in feature_columns if col in self.data.columns]
        
        if not available_features:
            print("没有可用的特征列用于推荐系统")
            return pd.DataFrame()
        
        # 提取特征并进行归一化
        features = self.data[available_features]
        
        # 对非二进制特征进行归一化
        for col in features.columns:
            if col != 'mode' and col != 'key':  # 假设mode和key是分类特征
                min_val = features[col].min()
                max_val = features[col].max()
                if max_val - min_val > 0:
                    features[col] = (features[col] - min_val) / (max_val - min_val)
        
        return features
    
    def compute_similarity(self):
        """计算音乐之间的相似度矩阵"""
        features = self.prepare_features()
        
        if features.empty:
            return False
        
        # 计算余弦相似度
        self.similarity_matrix = cosine_similarity(features)
        
        # 创建曲目索引映射
        self.track_indices = pd.Series(self.data.index, index=self.data['track_name'])
        
        return True
    
    def recommend_similar_songs(self, track_name, n_recommendations=10):
        """基于内容的音乐推荐"""
        if self.similarity_matrix is None or self.track_indices is None:
            print("相似度矩阵未计算，请先调用compute_similarity方法")
            return pd.DataFrame()
        
        # 获取曲目索引
        if track_name not in self.track_indices:
            print(f"曲目 '{track_name}' 不在数据集中")
            return pd.DataFrame()
        
        idx = self.track_indices[track_name]
        
        # 获取该曲目的相似度分数
        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        
        # 按相似度排序
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # 获取前n个推荐
        sim_scores = sim_scores[1:n_recommendations+1]
        
        # 获取推荐曲目索引
        track_indices = [i[0] for i in sim_scores]
        
        # 返回推荐结果
        recommendations = self.data.iloc[track_indices][['track_name', 'artist', 'genre', 'popularity']]
        recommendations['similarity'] = [i[1] for i in sim_scores]
        
        return recommendations
    
    def generate_user_recommendations(self, user_history, n_recommendations=10):
        """基于用户历史的个性化推荐"""
        if self.similarity_matrix is None or self.track_indices is None:
            print("相似度矩阵未计算，请先调用compute_similarity方法")
            return pd.DataFrame()
        
        # 初始化用户偏好向量
        user_profile = np.zeros(self.similarity_matrix.shape[1])
        
        # 构建用户偏好
        for track_name in user_history:
            if track_name in self.track_indices:
                idx = self.track_indices[track_name]
                user_profile += self.similarity_matrix[idx]
        
        # 归一化用户偏好
        if np.sum(user_profile) > 0:
            user_profile = user_profile / len(user_history)
        
        # 计算用户与所有音乐的相似度
        user_sim_scores = list(enumerate(user_profile))
        
        # 排除已听过的音乐
        heard_indices = [self.track_indices[track] for track in user_history if track in self.track_indices]
        user_sim_scores = [score for score in user_sim_scores if score[0] not in heard_indices]
        
        # 按相似度排序
        user_sim_scores = sorted(user_sim_scores, key=lambda x: x[1], reverse=True)
        
        # 获取前n个推荐
        user_sim_scores = user_sim_scores[:n_recommendations]
        
        # 获取推荐曲目索引
        track_indices = [i[0] for i in user_sim_scores]
        
        # 返回推荐结果
        recommendations = self.data.iloc[track_indices][['track_name', 'artist', 'genre', 'popularity']]
        recommendations['similarity'] = [i[1] for i in user_sim_scores]
        
        return recommendations


if __name__ == "__main__":
    # 示例使用
    try:
        data = pd.read_csv("../data/music_data.csv")
        print(f"加载数据完成，共{len(data)}条记录")
        
        # 初始化推荐系统
        rs = MusicRecommendationSystem(data)
        
        # 计算相似度
        if rs.compute_similarity():
            # 基于内容的推荐示例
            print("\n基于内容的推荐示例:")
            similar_songs = rs.recommend_similar_songs("Shape of You", n_recommendations=5)
            print(similar_songs)
            
            # 保存推荐结果
            similar_songs.to_csv("../results/recommendations/content_based_recommendations.csv", index=False)
            
            # 基于用户历史的推荐示例
            print("\n基于用户历史的推荐示例:")
            user_history = ["Blinding Lights", "Dance Monkey", "Watermelon Sugar"]
            user_recommendations = rs.generate_user_recommendations(user_history, n_recommendations=5)
            print(user_recommendations)
            
            # 保存推荐结果
            user_recommendations.to_csv("../results/recommendations/user_based_recommendations.csv", index=False)
    except Exception as e:
        print(f"推荐系统运行出错: {e}")    
