import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 연령대 이름
AGE_LABELS = ['10s', '20s', '30s', '40s', '50s', '60s', '70s']
SPEND_COLS = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
AGE_PLOT_PATH = 'plot_age_transported.png'
DESTINATION_PLOT_PATH = 'plot_destination_age_distribution.png'


def load_data():
    # train, test 파일 읽기
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')

    # test에는 정답이 없으므로 빈 칸으로 맞춘 뒤 병합
    test_copy = test_df.copy()
    test_copy['Transported'] = np.nan
    merged_df = pd.concat([train_df, test_copy], ignore_index=True)
    return train_df, test_df, merged_df


def print_total_count(train_df, test_df, all_data):
    # 전체 데이터 개수 출력
    print(f'train 데이터 수: {len(train_df)}')
    print(f'test 데이터 수: {len(test_df)}')
    print(f'전체 데이터 수: {len(all_data)}')


def make_correlation_data(train_df):
    # 상관계수 계산에 사용할 데이터만 따로 준비하기
    data = train_df.copy()

    # True / False 값을 1 / 0으로 바꾸기
    data['Transported'] = data['Transported'].map({True: 1, False: 0})
    data['CryoSleep'] = data['CryoSleep'].map({True: 1, False: 0})
    data['VIP'] = data['VIP'].map({True: 1, False: 0})

    # 숫자로 계산해야 하므로 숫자형으로 바꾸기
    for col in ['Age'] + SPEND_COLS:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # 필요한 숫자형 컬럼만 선택하기
    numeric_data = data[
        ['Transported', 'CryoSleep', 'VIP', 'Age'] + SPEND_COLS
    ].copy()

    # 빈칸은 0으로 채우기
    numeric_data = numeric_data.fillna(0)
    return numeric_data


def find_most_related_feature(train_df):
    # Transported와 각 항목의 상관계수 계산
    numeric_data = make_correlation_data(train_df)
    corr = numeric_data.corr(numeric_only=True)['Transported'].drop(labels=['Transported'])
    corr = corr.abs().sort_values(ascending=False)

    # 가장 큰 값 하나만 꺼내기
    top_feature = corr.index[0]
    top_score = corr.iloc[0]
    return top_feature, top_score


def make_age_group(age):
    # 나이가 비어 있거나 범위를 벗어나면 제외하기
    if pd.isna(age) or age < 10 or age >= 80:
        return np.nan

    # 10으로 나눈 몫을 이용해서 연령대 이름 찾기
    group_index = int(age // 10) - 1
    return AGE_LABELS[group_index]


def prepare_age_group_data(data):
    # 나이를 숫자로 바꾸고 사용할 수 있는 행만 남기기
    data = data.copy()
    data['Age'] = pd.to_numeric(data['Age'], errors='coerce')
    data = data.dropna(subset=['Age'])

    # 나이를 연령대로 바꾸기
    data['AgeGroup'] = data['Age'].apply(make_age_group)
    data = data.dropna(subset=['AgeGroup'])
    return data


def make_age_graph_data(train_df):
    # 나이와 정답 데이터만 가져오기
    data = train_df[['Age', 'Transported']].copy()

    # 정답이 없는 행은 제거하기
    data = data.dropna(subset=['Age', 'Transported'])

    # 연령대 그래프에 맞는 형태로 정리하기
    data = prepare_age_group_data(data)

    # 연령대별, Transported별 인원 수 세기
    result = data.groupby(['AgeGroup', 'Transported']).size().unstack(fill_value=0)
    result = result.reindex(AGE_LABELS, fill_value=0)

    # 혹시 한쪽 값이 없으면 0으로 만들기
    if True not in result.columns:
        result[True] = 0
    if False not in result.columns:
        result[False] = 0

    return result


def age_graph(train_df, output_path):
    # 그래프에 사용할 표 만들기
    result = make_age_graph_data(train_df)

    # 막대 그래프 그리기
    x = np.arange(len(AGE_LABELS))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, result[True].to_numpy(), width=width, label='Transported = True')
    ax.bar(x + width / 2, result[False].to_numpy(), width=width, label='Transported = False')
    ax.set_title('Transported Count by Age Group')
    ax.set_xlabel('Age Group')
    ax.set_ylabel('Passenger Count')
    ax.set_xticks(x)
    ax.set_xticklabels(AGE_LABELS)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    fig.tight_layout()

    # 파일로 저장하기
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def make_destination_graph_data(train_df):
    # 목적지와 나이 데이터만 가져오기
    data = train_df[['Destination', 'Age']].copy()

    # 목적지 빈칸은 Unknown으로 바꾸기
    data['Destination'] = data['Destination'].fillna('Unknown')

    # 연령대 그래프에 맞는 형태로 정리하기
    data = prepare_age_group_data(data)

    # 목적지별 연령대 인원 수 세기
    result = data.groupby(['Destination', 'AgeGroup']).size().unstack(fill_value=0)
    result = result.reindex(columns=AGE_LABELS, fill_value=0)
    return result


def destination_graph(train_df, output_path):
    # 그래프에 사용할 표 만들기
    result = make_destination_graph_data(train_df)

    # 누적 막대 그래프 그리기
    fig, ax = plt.subplots(figsize=(10, 6))
    result.plot(kind='bar', stacked=True, ax=ax)
    ax.set_title('Age Distribution by Destination')
    ax.set_xlabel('Destination')
    ax.set_ylabel('Number of Passengers')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend(title='Age Group', bbox_to_anchor=(1.02, 1), loc='upper left')
    fig.tight_layout()

    # 파일로 저장하기
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_graphs(train_df):
    # 그래프 2개를 파일로 저장하기
    age_graph(train_df, AGE_PLOT_PATH)
    destination_graph(train_df, DESTINATION_PLOT_PATH)


def main():
    # 1. 데이터 읽기
    train_df, test_df, merged_df = load_data()

    # 2. 전체 데이터 수 확인
    print_total_count(train_df, test_df, merged_df)

    # 3. 어떤 항목이 가장 관련이 큰지 확인
    top_feature, top_score = find_most_related_feature(train_df)
    print(f'Transported와 가장 관련이 큰 항목: {top_feature} ({top_score:.4f})')

    # 4. 그래프 2개 저장
    save_graphs(train_df)


if __name__ == '__main__':
    main()
