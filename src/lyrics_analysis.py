import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os
import re

# 下载必要的NLTK数据
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')

# 创建结果目录
os.makedirs('../results/lyrics_analysis', exist_ok=True)

class LyricsAnalyzer:
    def __init__(self, data):
        """初始化歌词分析器"""
        self.data = data
        self.sid = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))
        
    def clean_lyrics(self, text):
        """清洗歌词文本"""
        if not isinstance(text, str):
            return ""
        
        # 转换为小写
        text = text.lower()
        
        # 移除特殊字符和数字
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # 分词
        tokens = word_tokenize(text)
        
        # 移除停用词
        filtered_tokens = [word for word in tokens if word not in self.stop_words]
        
        # 重新组合
        return " ".join(filtered_tokens)
    
    def analyze_sentiment(self, text):
        """分析歌词情感"""
        if not text:
            return {'neg': 0.0, 'neu': 0.0, 'pos': 0.0, 'compound': 0.0}
        
        return self.sid.polarity_scores(text)
    
    def process_lyrics(self):
        """处理所有歌词并添加情感分析结果"""
        if 'lyrics' not in self.data.columns:
            print("数据集中没有'lyrics'列")
            return False
        
        # 清洗歌词
        self.data['cleaned_lyrics'] = self.data['lyrics'].apply(self.clean_lyrics)
        
        # 分析情感
        sentiment_scores = self.data['cleaned_lyrics'].apply(self.analyze_sentiment)
        
        # 提取情感分数
        self.data['negative_sentiment'] = sentiment_scores.apply(lambda x: x['neg'])
        self.data['neutral_sentiment'] = sentiment_scores.apply(lambda x: x['neu'])
        self.data['positive_sentiment'] = sentiment_scores.apply(lambda x: x['pos'])
        self.data['compound_sentiment'] = sentiment_scores.apply(lambda x: x['compound'])
        
        return True
    
    def generate_wordcloud(self, genre=None):
        """生成词云"""
        if 'cleaned_lyrics' not in self.data.columns:
            print("请先调用process_lyrics方法")
            return
        
        # 根据流派筛选
        if genre:
            genre_data = self.data[self.data['genre'] == genre]
            if genre_data.empty:
                print(f"未找到流派为'{genre}'的歌曲")
                return
            text = " ".join(genre_data['cleaned_lyrics'].dropna())
            title = f"{genre}流派歌词词云"
            filename = f"wordcloud_{genre}.png"
        else:
            text = " ".join(self.data['cleaned_lyrics'].dropna())
            title = "所有歌词词云"
            filename = "wordcloud_all.png"
        
        if not text:
            print("没有可用的歌词文本")
            return
        
        # 生成词云
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        
        # 显示词云图
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(f'../results/lyrics_analysis/{filename}')
        plt.close()
    
    def analyze_sentiment_by_genre(self):
        """按流派分析平均情感"""
        if 'compound_sentiment' not in self.data.columns or 'genre' not in self.data.columns:
            print("数据不包含必要的列")
            return pd.DataFrame()
        
        # 按流派计算平均情感
        genre_sentiment = self.data.groupby('genre').agg({
            'compound_sentiment': 'mean',
            'positive_sentiment': 'mean',
            'negative_sentiment': 'mean',
            'neutral_sentiment': 'mean'
        }).reset_index()
        
        # 按复合情感排序
        genre_sentiment = genre_sentiment.sort_values('compound_sentiment', ascending=False)
        
        # 保存结果
        genre_sentiment.to_csv('../results/lyrics_analysis/genre_sentiment.csv', index=False)
        
        # 可视化
        plt.figure(figsize=(12, 6))
        sns.barplot(x='compound_sentiment', y='genre', data=genre_sentiment)
        plt.title('各流派平均情感分数')
        plt.xlabel('复合情感分数')
        plt.tight_layout()
        plt.savefig('../results/lyrics_analysis/genre_sentiment.png')
        plt.close()
        
        return genre_sentiment


if __name__ == "__main__":
    # 示例使用
    try:
        data = pd.read_csv("../data/music_data.csv")
        print(f"加载数据完成，共{len(data)}条记录")
        
        # 初始化歌词分析器
        analyzer = LyricsAnalyzer(data)
        
        # 处理歌词
        if analyzer.process_lyrics():
            print("\n歌词处理完成")
            
            # 保存处理后的数据
            analyzer.data.to_csv("../data/music_data_with_sentiment.csv", index=False)
            
            # 生成词云
            print("生成词云...")
            analyzer.generate_wordcloud()  # 全部歌词词云
            analyzer.generate_wordcloud(genre="Pop")
            analyzer.generate_wordcloud(genre="Rock")
            
            # 分析流派情感
            print("分析流派情感...")
            genre_sentiment = analyzer.analyze_sentiment_by_genre()
            print("\n各流派平均情感分数:")
            print(genre_sentiment)
    except Exception as e:
        print(f"歌词分析出错: {e}")    
