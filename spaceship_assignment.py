import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 나이 구간 라벨
age_group_labels = ['10s', '20s', '30s', '40s', '50s', '60s', '70s']

# 나이 구간 기준
age_bins = [10, 20, 30, 40, 50, 60, 70, 80]


def read_datasets():
    """train.csv와 test.csv를 읽는다."""
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')

    return train_df, test_df


def merge_datasets(train_df, test_df):
    """train과 test를 하나로 합친다."""
    train_copy = train_df.copy()
    test_copy = test_df.copy()

    train_copy['Source'] = 'train'
    test_copy['Source'] = 'test'
    test_copy['Transported'] = np.nan

    merged_df = pd.concat([train_copy, test_copy], axis=0, ignore_index=True)
    return merged_df


def add_analysis_features(df):
    """분석에 필요한 파생 변수를 만든다."""
    result = df.copy()

    # PassengerId에서 그룹 번호 추출
    group_id = result['PassengerId'].astype(str).str.split('_', n=1).str[0]
    result['GroupNumber'] = pd.to_numeric(group_id, errors='coerce')

    # Cabin에서 갑판과 방향 추출
    cabin_parts = result['Cabin'].fillna('Unknown/0/U').str.split('/', expand=True)
    result['CabinDeck'] = cabin_parts[0]
    result['CabinSide'] = cabin_parts[2]

    # 숫자형 데이터로 변환
    numeric_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    for col in numeric_cols:
        result[col] = pd.to_numeric(result[col], errors='coerce')

    # 편의시설 총 지출 계산
    result['TotalSpend'] = (
        result['RoomService'].fillna(0.0)
        + result['FoodCourt'].fillna(0.0)
        + result['ShoppingMall'].fillna(0.0)
        + result['Spa'].fillna(0.0)
        + result['VRDeck'].fillna(0.0)
    )

    result['CryoSleep'] = result['CryoSleep'].fillna(False).astype(int)
    result['VIP'] = result['VIP'].fillna(False).astype(int)

    return result


def find_most_related_feature(train_df):
    """Transported와 가장 관련이 큰 항목을 찾는다."""
    target = train_df['Transported'].astype(int)
    features = add_analysis_features(train_df)

    # 비교할 변수들 선택
    feature_list = [
        'HomePlanet',
        'CryoSleep',
        'Destination',
        'Age',
        'VIP',
        'RoomService',
        'FoodCourt',
        'ShoppingMall',
        'Spa',
        'VRDeck',
        'TotalSpend',
        'CabinDeck',
        'CabinSide',
        'GroupNumber',
    ]

    work = features[feature_list].copy()

    numeric_features = [
        'Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
        'TotalSpend', 'GroupNumber',
    ]
    categorical_features = ['HomePlanet', 'Destination', 'CabinDeck', 'CabinSide']

    # 숫자형 결측값은 중앙값으로 채움
    for col in numeric_features:
        median_value = work[col].median()
        if np.isnan(median_value):
            median_value = 0.0
        work[col] = work[col].fillna(median_value)

    # 문자형 결측값은 Unknown으로 채움
    for col in categorical_features:
        work[col] = work[col].fillna('Unknown')

    # 문자형 데이터를 숫자로 바꿈
    encoded = pd.get_dummies(
        work,
        columns=categorical_features,
        prefix_sep='__',
        dtype=float,
    )
    encoded['Transported'] = target.to_numpy()

    # Transported와의 상관계수 계산
    corr_series = encoded.corr(numeric_only=True)['Transported'].drop(labels=['Transported'])

    feature_scores = {}
    for raw_feature_name, score in corr_series.items():
        feature_name = str(raw_feature_name)

        # 원-핫 인코딩된 이름을 원래 변수 이름으로 묶음
        if '__' in feature_name:
            base_name = feature_name.split('__', maxsplit=1)[0]
        else:
            base_name = feature_name

        current_max = feature_scores.get(base_name, 0.0)
        abs_value = float(abs(score))

        if abs_value > current_max:
            feature_scores[base_name] = abs_value

    sorted_items = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
    top_feature, top_score = sorted_items[0]

    return top_feature, top_score


def make_age_group_series(age_series):
    """나이를 10세 단위 구간으로 나눈다."""
    age_group = pd.cut(
        age_series,
        bins=age_bins,
        labels=age_group_labels,
        right=False,
        include_lowest=False,
    )
    return age_group


def plot_transported_by_age_group(train_df, output_path):
    """연령대별 Transported 그래프를 저장한다."""
    work = train_df[['Age', 'Transported']].copy()
    work['Age'] = pd.to_numeric(work['Age'], errors='coerce')
    work = work.dropna(subset=['Age', 'Transported'])

    # 나이를 연령대로 변환
    work['AgeGroup'] = make_age_group_series(work['Age'])
    work = work.dropna(subset=['AgeGroup'])

    # 연령대와 Transported 기준으로 개수 계산
    count_table = (
        work.groupby(['AgeGroup', 'Transported'], observed=False)
        .size()
        .unstack(fill_value=0)
        .reindex(age_group_labels)
    )

    if True not in count_table.columns:
        count_table[True] = 0
    if False not in count_table.columns:
        count_table[False] = 0

    x = np.arange(len(age_group_labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 5))

    # True / False 막대를 나란히 그림
    ax.bar(
        x - width / 2,
        count_table[True].to_numpy(),
        width=width,
        label='Transported = True',
        color='#4ECDC4',
    )

    ax.bar(
        x + width / 2,
        count_table[False].to_numpy(),
        width=width,
        label='Transported = False',
        color='#FF6B6B',
    )

    ax.set_title('Transported Count by Age Group', fontsize=14, fontweight='bold')
    ax.set_xlabel('Age Group')
    ax.set_ylabel('Passenger Count')
    ax.set_xticks(x)
    ax.set_xticklabels(age_group_labels)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_destination_age_distribution(train_df, output_path):
    """Destination별 연령대 비율 그래프를 저장한다."""
    work = train_df[['Destination', 'Age']].copy()
    work['Age'] = pd.to_numeric(work['Age'], errors='coerce')
    work['Destination'] = work['Destination'].fillna('Unknown')
    work = work.dropna(subset=['Age'])

    # 나이를 연령대로 변환
    work['AgeGroup'] = make_age_group_series(work['Age'])
    work = work.dropna(subset=['AgeGroup'])

    # 목적지별 연령대 인원수 계산
    count_table = (
        work.groupby(['Destination', 'AgeGroup'], observed=False)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=age_group_labels, fill_value=0)
    )

    # 목적지별 비율로 변환
    rate_table = count_table.div(
        count_table.sum(axis=1).replace(0, np.nan),
        axis=0,
    ).fillna(0.0)

    fig, ax = plt.subplots(figsize=(11, 6))
    bottom = np.zeros(len(rate_table))

    # 연령대마다 다른 색으로 누적 막대 생성
    colors = plt.get_cmap('viridis')(np.linspace(0, 1, len(age_group_labels)))

    for idx, age_group in enumerate(age_group_labels):
        values = rate_table[age_group].to_numpy()
        ax.bar(
            rate_table.index,
            values,
            bottom=bottom,
            label=age_group,
            color=colors[idx],
        )
        bottom += values

    ax.set_title('Age Group Ratio by Destination', fontsize=14, fontweight='bold')
    ax.set_xlabel('Destination')
    ax.set_ylabel('Ratio (0 to 1)')
    ax.set_ylim(0, 1)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    ax.legend(
        title='Age Group',
        bbox_to_anchor=(1.02, 1),
        loc='upper left',
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    """전체 분석 과정을 실행한다."""
    print('Spaceship Titanic 분석 시작')
    print('-' * 40)

    # 데이터 읽기와 병합
    train_df, test_df = read_datasets()
    merged_df = merge_datasets(train_df, test_df)

    # 데이터 개수 출력
    print(f'train 데이터 수: {len(train_df)}')
    print(f'test 데이터 수: {len(test_df)}')
    print(f'전체 데이터 수: {len(merged_df)}')

    # 가장 관련이 큰 항목 출력
    top_feature, top_score = find_most_related_feature(train_df)
    print(f'Transported와 가장 관련이 큰 항목: {top_feature} ({top_score:.4f})')

    # 그래프 1 저장
    graph1_path = 'plot_age_transported.png'
    plot_transported_by_age_group(train_df, graph1_path)
    print(f'그래프 저장 완료: {graph1_path}')

    # 그래프 2 저장
    graph2_path = 'plot_destination_age_distribution.png'
    plot_destination_age_distribution(train_df, graph2_path)
    print(f'그래프 저장 완료: {graph2_path}')
    print('-' * 40)
    print('분석 완료')

if __name__ == '__main__':
    main()
