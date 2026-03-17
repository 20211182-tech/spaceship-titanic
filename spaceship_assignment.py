import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 연령대 라벨 (10세 단위)
AGE_LABELS = ['10s', '20s', '30s', '40s', '50s', '60s', '70s']
# 연령대 구간 경계
AGE_BINS = [10, 20, 30, 40, 50, 60, 70, 80]


def read_datasets():
    # train/test 파일 읽기
    return pd.read_csv('train.csv'), pd.read_csv('test.csv')


def merge_datasets(train_df, test_df):
    # 원본 보존을 위해 복사해서 작업
    train_copy = train_df.copy()
    test_copy = test_df.copy()

    # 데이터 출처 표시
    train_copy['Source'] = 'train'
    test_copy['Source'] = 'test'

    # test에는 정답이 없으므로 빈값으로 생성
    test_copy['Transported'] = np.nan

    # 행 기준으로 이어붙여 전체 데이터 생성
    return pd.concat([train_copy, test_copy], axis=0, ignore_index=True)


def add_analysis_features(df):
    # 분석에 필요한 파생 변수 생성
    result = df.copy()

    # PassengerId에서 그룹 번호 추출
    group_id = result['PassengerId'].astype(str).str.split('_', n=1).str[0]
    result['GroupNumber'] = pd.to_numeric(group_id, errors='coerce')

    # Cabin에서 Deck / Side 분리
    cabin_parts = result['Cabin'].fillna('Unknown/0/U').str.split('/', expand=True)
    result['CabinDeck'] = cabin_parts[0]
    result['CabinSide'] = cabin_parts[2]

    # 수치형 컬럼 정리
    spend_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    for col in ['Age'] + spend_cols:
        result[col] = pd.to_numeric(result[col], errors='coerce')

    # 편의시설 총 지출
    result['TotalSpend'] = sum(result[col].fillna(0.0) for col in spend_cols)
    # 불리언을 0/1로 변환
    result['CryoSleep'] = result['CryoSleep'].fillna(False).astype(int)
    result['VIP'] = result['VIP'].fillna(False).astype(int)
    return result


def find_most_related_feature(train_df):
    # 정답을 숫자형으로 변환
    target = train_df['Transported'].astype(int)

    # 비교할 변수들만 선택
    work = add_analysis_features(train_df)[[
        'HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP',
        'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
        'TotalSpend', 'CabinDeck', 'CabinSide', 'GroupNumber',
    ]].copy()

    numeric_cols = [
        'Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa',
        'VRDeck', 'TotalSpend', 'GroupNumber',
    ]
    categorical_cols = ['HomePlanet', 'Destination', 'CabinDeck', 'CabinSide']

    # 결측값 처리
    for col in numeric_cols:
        median_value = work[col].median()
        work[col] = work[col].fillna(0.0 if np.isnan(median_value) else median_value)
    for col in categorical_cols:
        work[col] = work[col].fillna('Unknown')

    # 문자형 데이터를 원-핫 인코딩
    encoded = pd.get_dummies(work, columns=categorical_cols, prefix_sep='__', dtype=float)
    encoded['Transported'] = target.to_numpy()

    # Transported와의 상관계수 계산
    corr_series = encoded.corr(numeric_only=True)['Transported'].drop(labels=['Transported'])

    # 같은 원본 변수 기준으로 최대 절대 상관만 사용
    feature_scores = {}
    for raw_name, score in corr_series.items():
        name = str(raw_name)
        base_name = name.split('__', maxsplit=1)[0] if '__' in name else name
        feature_scores[base_name] = max(feature_scores.get(base_name, 0.0), float(abs(score)))

    # 가장 관련이 큰 변수 반환
    top_feature, top_score = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[0]
    return top_feature, top_score


def make_age_group_series(age_series):
    # 나이를 10세 단위 구간으로 자르기
    return pd.cut(age_series, bins=AGE_BINS, labels=AGE_LABELS, right=False, include_lowest=False)


def plot_transported_by_age_group(train_df, output_path):
    # 연령대 + Transported 조합별 인원 집계
    work = train_df[['Age', 'Transported']].copy()
    work['Age'] = pd.to_numeric(work['Age'], errors='coerce')
    work = work.dropna(subset=['Age', 'Transported'])
    work['AgeGroup'] = make_age_group_series(work['Age'])
    work = work.dropna(subset=['AgeGroup'])

    count_table = (
        work.groupby(['AgeGroup', 'Transported'], observed=False)
        .size().unstack(fill_value=0).reindex(AGE_LABELS)
    )

    # 특정 클래스가 없을 때 대비
    if True not in count_table.columns:
        count_table[True] = 0
    if False not in count_table.columns:
        count_table[False] = 0

    # 막대 그래프 그리기
    x = np.arange(len(AGE_LABELS))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, count_table[True].to_numpy(), width=width, label='Transported = True', color='#4ECDC4')
    ax.bar(x + width / 2, count_table[False].to_numpy(), width=width, label='Transported = False', color='#FF6B6B')
    ax.set_title('Transported Count by Age Group', fontsize=14, fontweight='bold')
    ax.set_xlabel('Age Group')
    ax.set_ylabel('Passenger Count')
    ax.set_xticks(x)
    ax.set_xticklabels(AGE_LABELS)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_destination_age_distribution(train_df, output_path):
    # 목적지별 연령대 분포 집계
    work = train_df[['Destination', 'Age']].copy()
    work['Age'] = pd.to_numeric(work['Age'], errors='coerce')
    work['Destination'] = work['Destination'].fillna('Unknown')
    work = work.dropna(subset=['Age'])
    work['AgeGroup'] = make_age_group_series(work['Age'])
    work = work.dropna(subset=['AgeGroup'])

    count_table = (
        work.groupby(['Destination', 'AgeGroup'], observed=False)
        .size().unstack(fill_value=0).reindex(columns=AGE_LABELS, fill_value=0)
    )

    # 누적 막대 그래프로 시각화
    fig, ax = plt.subplots(figsize=(10, 6))
    count_table.plot(kind='bar', stacked=True, colormap='viridis', ax=ax)

    ax.set_title('Age Distribution by Destination', fontsize=14, fontweight='bold')
    ax.set_xlabel('Destination')
    ax.set_ylabel('Number of Passengers')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend(title='Age Group', bbox_to_anchor=(1.02, 1), loc='upper left')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    # 전체 실행 흐름
    print('Spaceship Titanic 분석 시작')
    print('-' * 40)

    # 1) 데이터 읽기 및 병합
    train_df, test_df = read_datasets()
    merged_df = merge_datasets(train_df, test_df)
    print(f'train 데이터 수: {len(train_df)}')
    print(f'test 데이터 수: {len(test_df)}')
    print(f'전체 데이터 수: {len(merged_df)}')

    # 2) 관련성 가장 큰 변수 찾기
    top_feature, top_score = find_most_related_feature(train_df)
    print(f'Transported와 가장 관련이 큰 항목: {top_feature} ({top_score:.4f})')

    # 3) 그래프 2개 저장
    graph1_path = 'plot_age_transported.png'
    graph2_path = 'plot_destination_age_distribution.png'
    plot_transported_by_age_group(train_df, graph1_path)
    plot_destination_age_distribution(train_df, graph2_path)
    print(f'그래프 저장 완료: {graph1_path}')
    print(f'그래프 저장 완료: {graph2_path}')
    print('-' * 40)
    print('분석 완료')


if __name__ == '__main__':
    main()
