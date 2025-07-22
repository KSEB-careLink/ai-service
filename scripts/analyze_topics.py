import json
import os

def calculate_topic_stats(dataset_path, results_path, output_dir):
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    id_to_topic = {item['question_id']: item['topic'] for item in dataset}

    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    topic_stats = {}
    for record in results:
        qid = record['question_id']
        topic = id_to_topic.get(qid)
        if topic is None:
            continue
        if topic not in topic_stats:
            topic_stats[topic] = {'count': 0, 'correct_count': 0, 'time_sum': 0}
        topic_stats[topic]['count'] += 1
        if record['correct']:
            topic_stats[topic]['correct_count'] += 1
        topic_stats[topic]['time_sum'] += record['time_taken']

    for topic, stats in topic_stats.items():
        stats['accuracy'] = stats['correct_count'] / stats['count']
        stats['avg_time'] = stats['time_sum'] / stats['count']

    difficult_topics = [
        t for t, s in topic_stats.items()
        if s['accuracy'] < 0.6 or s['avg_time'] > 6
    ]

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'topic_stats.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "topic_stats": topic_stats,
            "difficult_topics": difficult_topics
        }, f, ensure_ascii=False, indent=2)

    print(f"✅ 결과가 {output_path} 에 저장되었습니다!")
    return topic_stats, difficult_topics


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))  # scripts 폴더
    dataset_path = os.path.join(base_dir, '..', 'data', 'dataset.json')
    results_path = os.path.join(base_dir, '..', 'data', 'results.json')
    output_dir = os.path.join(base_dir, '..', 'models')

    dataset_path = os.path.normpath(dataset_path)
    results_path = os.path.normpath(results_path)
    output_dir = os.path.normpath(output_dir)

    print(f"[DEBUG] dataset_path = {dataset_path}")
    print(f"[DEBUG] results_path = {results_path}")
    print(f"[DEBUG] output_dir = {output_dir}")

    calculate_topic_stats(dataset_path, results_path, output_dir)
