import argparse
from data_analysis import load_data, exploratory_analysis, train_genre_classifier
from recommendation_system import MusicRecommendationSystem
from lyrics_analysis import LyricsAnalyzer

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='音乐数据分析与推荐系统')
    parser.add_argument('--data', type=str, default='../data/music_data.csv', help='数据文件路径')
    parser.add_argument('--analyze', action='store_true', help='执行数据分析')
    parser.add_argument('--recommend', action='store_true', help='执行推荐系统')
    parser.add_argument('--lyrics', action='store_true', help='执行歌词分析')
    args = parser.parse_args()
    
    print("===== 音乐数据分析项目 =====")
    
    # 加载数据
    data = load_data(args.data)
    
    if data.empty:
        print("数据加载失败，程序退出")
        return
    
    # 执行数据分析
    if args.analyze:
        print("\n=== 执行探索性数据分析 ===")
        exploratory_analysis(data)
        
        print("\n=== 训练流派分类模型 ===")
        train_genre_classifier(data)
    
    # 执行推荐系统
    if args.recommend:
        print("\n=== 执行推荐系统 ===")
        rs = MusicRecommendationSystem(data)
        
        if rs.compute_similarity():
            print("\n基于内容的推荐示例:")
            similar_songs = rs.recommend_similar_songs("Shape of You", n_recommendations=5)
            print(similar_songs)
            
            print("\n基于用户历史的推荐示例:")
            user_history = ["Blinding Lights", "Dance Monkey", "Watermelon Sugar"]
            user_recommendations = rs.generate_user_recommendations(user_history, n_recommendations=5)
            print(user_recommendations)
    
    # 执行歌词分析
    if args.lyrics:
        print("\n=== 执行歌词分析 ===")
        analyzer = LyricsAnalyzer(data)
        
        if analyzer.process_lyrics():
            print("\n歌词处理完成")
            
            # 生成词云
            analyzer.generate_wordcloud()
            analyzer.generate_wordcloud(genre="Pop")
            analyzer.generate_wordcloud(genre="Rock")
            
            # 分析流派情感
            genre_sentiment = analyzer.analyze_sentiment_by_genre()
            print("\n各流派平均情感分数:")
            print(genre_sentiment)
    
    print("\n程序执行完毕！")

if __name__ == "__main__":
    main()    
