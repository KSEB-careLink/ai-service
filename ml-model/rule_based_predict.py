def rule_based_predict(logs):
    """
    최근 푼 문제 기록을 기반으로 간단한 정답률 추론
    logs: [{'is_correct': 1, 'duration': 8.5, 'solved_date': '2025-07-30'}, ...]
    """
    if not logs or len(logs) < 3:
        return 0.5  

    weights = [1.5 - i * 0.05 for i in range(len(logs))]  
    accs = [log['is_correct'] for log in logs]
    times = [log['duration'] for log in logs]

    total_w = sum(weights)
    weighted_acc = sum(a * w for a, w in zip(accs, weights)) / total_w
    weighted_time = sum(t * w for t, w in zip(times, weights)) / total_w

    if weighted_acc >= 0.8 and weighted_time < 10:
        return min(weighted_acc + 0.05, 0.95)
    elif weighted_acc <= 0.5:
        return max(weighted_acc - 0.05, 0.4)
    else:
        return weighted_acc


def get_predicted_accuracy(logs, model):
    """
    사용자 풀이 기록(logs)에 따라 예측 방식을 분기 처리
    logs: [{'is_correct': 1, 'duration': 9.2, 'solved_date': '2025-07-31'}, ...]
    model: 학습된 sklearn 모델 객체 (joblib.load 등으로 전달)
    """
    if not logs:
        return 0.5 

    solved_days = {log['solved_date'] for log in logs}
    num_days = len(solved_days)

    if num_days < 10:
        return rule_based_predict(logs)

    elif num_days < 90:
        avg_acc = sum(log['is_correct'] for log in logs) / len(logs)
        avg_time = sum(log['duration'] for log in logs) / len(logs)
        return model.predict([[avg_acc, avg_time]])[0]

    else:
        sorted_logs = sorted(logs, key=lambda x: x['solved_date'])[-90:]
        avg_acc = sum(log['is_correct'] for log in sorted_logs) / 90
        avg_time = sum(log['duration'] for log in sorted_logs) / 90
        return model.predict([[avg_acc, avg_time]])[0]
