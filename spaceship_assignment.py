"""
Spaceship Titanic - 데이터 탐색 및 시각화 과제

수행 내용:
    1. train.csv, test.csv 읽기 및 병합
    2. 전체 데이터 수량 파악
    3. Transported와 가장 관련성이 높은 항목 탐색
    4. 연령대별 Transported 현황 시각화
    5. (보너스) Destination별 연령대 분포 시각화

사용 라이브러리: pandas, numpy, matplotlib
"""

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# 연령대 레이블 (10세 단위, 10대~70대)
AGE_GROUP_LABELS = ['10s', '20s', '30s', '40s', '50s', '60s', '70s']

# pd.cut 에 사용할 구간 경계값 (right=False 이므로 왼쪽 포함, 오른쪽 미포함)
AGE_BINS = [10, 20, 30, 40, 50, 60, 70, 80]


def read_datasets(base_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """train.csv 와 test.csv 를 읽어 DataFrame 으로 반환한다.

    Args:
        base_path: CSV 파일이 위치한 디렉터리 경로.

    Returns:
        (train_df, test_df) 튜플.
    """
    train_path = base_path / 'train.csv'
    test_path = base_path / 'test.csv'

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    return train_df, test_df


def merge_datasets(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """train / test 데이터를 하나의 DataFrame 으로 병합한다.

    - 원본 DataFrame 이 변경되지 않도록 복사본을 사용한다.
    - 출처를 구분하기 위해 'Source' 열을 추가한다.
    - test 데이터에는 Transported 정보가 없으므로 NaN 으로 채운다.

    Args:
        train_df: 학습용 데이터.
        test_df:  테스트용 데이터.

    Returns:
        병합된 전체 DataFrame.
    """
    train_copy = train_df.copy()
    test_copy = test_df.copy()

    # 각 행이 어느 출처에서 왔는지 표시
    train_copy['Source'] = 'train'
    test_copy['Source'] = 'test'

    # test 에는 정답 레이블이 없으므로 NaN 처리
    test_copy['Transported'] = np.nan

    merged_df = pd.concat([train_copy, test_copy], axis=0, ignore_index=True)
    return merged_df


def add_analysis_features(df: pd.DataFrame) -> pd.DataFrame:
    """분석에 필요한 파생 피처를 추가한다.

    추가 항목:
        - GroupNumber : PassengerId 에서 추출한 그룹 번호
        - CabinDeck   : Cabin 에서 추출한 갑판 정보 (A~T)
        - CabinSide   : Cabin 에서 추출한 좌현(P)/우현(S) 정보
        - TotalSpend  : 5개 편의시설 지출액의 합계

    Args:
        df: 원본 DataFrame (Transported 열 포함 가능).

    Returns:
        파생 피처가 추가된 새 DataFrame.
    """
    result = df.copy()

    # PassengerId 형식: <그룹번호>_<순번> → 그룹 번호만 추출
    group_id = result['PassengerId'].astype(str).str.split('_', n=1).str[0]
    result['GroupNumber'] = pd.to_numeric(group_id, errors='coerce')

    # Cabin 형식: <Deck>/<Num>/<Side> → 갑판과 좌·우현만 사용
    cabin_parts = result['Cabin'].fillna('Unknown/0/U').str.split('/', expand=True)
    result['CabinDeck'] = cabin_parts[0]
    result['CabinSide'] = cabin_parts[2]

    # 숫자형 열은 문자열 혼입 가능성이 있으므로 강제 변환
    numeric_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    for col in numeric_cols:
        result[col] = pd.to_numeric(result[col], errors='coerce')

    # 편의시설 5개 지출의 합 (결측은 0으로 처리)
    result['TotalSpend'] = (
        result['RoomService'].fillna(0.0)
        + result['FoodCourt'].fillna(0.0)
        + result['ShoppingMall'].fillna(0.0)
        + result['Spa'].fillna(0.0)
        + result['VRDeck'].fillna(0.0)
    )

    # 불리언 열은 정수(0/1)로 변환하여 상관계수 계산에 활용
    result['CryoSleep'] = result['CryoSleep'].fillna(False).astype(int)
    result['VIP'] = result['VIP'].fillna(False).astype(int)

    return result


def find_most_related_feature(train_df: pd.DataFrame) -> Tuple[str, float]:
    """Transported 와 상관관계가 가장 높은 피처를 찾는다.

    범주형 피처는 원-핫 인코딩 후 각 더미 열의 절대 상관계수 중
    최댓값을 해당 원본 피처의 점수로 사용한다.

    Args:
        train_df: 정답 레이블(Transported)이 포함된 학습 데이터.

    Returns:
        (최고 관련 피처명, 절대 상관계수) 튜플.
    """
    # Transported 를 0/1 정수로 변환하여 상관계수 계산 기준으로 사용
    target = train_df['Transported'].astype(int)
    features = add_analysis_features(train_df)

    selected_cols = [
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

    work = features[selected_cols].copy()

    # 수치형 / 범주형 구분
    numeric_cols = [
        'Age',
        'RoomService',
        'FoodCourt',
        'ShoppingMall',
        'Spa',
        'VRDeck',
        'TotalSpend',
        'GroupNumber',
    ]
    categorical_cols = ['HomePlanet', 'Destination', 'CabinDeck', 'CabinSide']

    # 수치형 결측값 → 중앙값으로 대체
    for col in numeric_cols:
        median_value = work[col].median()
        if np.isnan(median_value):
            median_value = 0.0
        work[col] = work[col].fillna(median_value)

    # 범주형 결측값 → 'Unknown' 문자열로 대체
    for col in categorical_cols:
        work[col] = work[col].fillna('Unknown')

    # 범주형 열을 원-핫 인코딩하여 상관계수 계산 가능한 형태로 변환
    encoded = pd.get_dummies(
        work,
        columns=categorical_cols,
        prefix_sep='__',
        dtype=float,
    )
    encoded['Transported'] = target.to_numpy()

    # Transported 와의 상관계수만 추출 (자기 자신 열 제외)
    corr_series = encoded.corr(numeric_only=True)['Transported'].drop(labels=['Transported'])

    # 원-핫 더미 열(예: HomePlanet__Earth)을 원본 피처 이름으로 집계
    # 같은 피처의 더미 열 중 절대 상관계수가 가장 큰 값을 대표 점수로 사용
    feature_scores: Dict[str, float] = {}
    for raw_feature_name, value in corr_series.items():
        feature_name = str(raw_feature_name)

        # '__' 구분자가 있으면 원-핫 더미 열 → 원본 피처명 복원
        if '__' in feature_name:
            base_name = feature_name.split('__', maxsplit=1)[0]
        else:
            base_name = feature_name

        current = feature_scores.get(base_name, 0.0)
        abs_value = float(abs(value))
        if abs_value > current:
            feature_scores[base_name] = abs_value

    # 점수 내림차순 정렬 후 1위 피처 반환
    sorted_items = sorted(feature_scores.items(), key=lambda item: item[1], reverse=True)
    top_feature, top_score = sorted_items[0]

    return top_feature, top_score


def make_age_group_series(age_series: pd.Series) -> pd.Series:
    """나이 시리즈를 10세 단위 연령대 레이블로 변환한다.

    AGE_BINS / AGE_GROUP_LABELS 전역 상수를 기준으로 구간을 나눈다.
    구간에 해당하지 않는 값(예: 10세 미만, 80세 이상)은 NaN 으로 처리된다.

    Args:
        age_series: 나이(숫자형) Series.

    Returns:
        연령대 레이블 Categorical Series.
    """
    age_group = pd.cut(
        age_series,
        bins=AGE_BINS,
        labels=AGE_GROUP_LABELS,
        right=False,       # 왼쪽 포함, 오른쪽 미포함 구간 [10, 20)
        include_lowest=False,
    )
    return age_group


def plot_transported_by_age_group(train_df: pd.DataFrame, output_path: Path) -> None:
    """연령대별 Transported 인원을 묶음 막대 그래프로 저장한다.

    True / False 두 값을 나란히 배치하여 연령대별 차이를 한눈에 파악할 수 있다.

    Args:
        train_df:    학습 데이터 (Age, Transported 열 포함).
        output_path: 그래프 이미지 저장 경로.
    """
    # 필요한 열만 선택하고 결측 행 제거
    work = train_df[['Age', 'Transported']].copy()
    work['Age'] = pd.to_numeric(work['Age'], errors='coerce')
    work = work.dropna(subset=['Age', 'Transported'])

    # 나이를 10세 단위 연령대로 변환 후 구간 외 데이터 제거
    work['AgeGroup'] = make_age_group_series(work['Age'])
    work = work.dropna(subset=['AgeGroup'])

    # 연령대 × Transported 크로스탭 생성 (AGE_GROUP_LABELS 순서 보장)
    count_table = (
        work.groupby(['AgeGroup', 'Transported'], observed=False)
        .size()
        .unstack(fill_value=0)
        .reindex(AGE_GROUP_LABELS)
    )

    # 데이터 편향으로 한쪽 열이 없을 경우 0으로 보완
    if True not in count_table.columns:
        count_table[True] = 0
    if False not in count_table.columns:
        count_table[False] = 0

    # 막대 위치 및 너비 설정
    x = np.arange(len(AGE_GROUP_LABELS))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        x - width / 2,
        count_table[True].to_numpy(),
        width=width,
        label='Transported = True',
    )
    ax.bar(
        x + width / 2,
        count_table[False].to_numpy(),
        width=width,
        label='Transported = False',
    )

    ax.set_title('Transported Count by Age Group')
    ax.set_xlabel('Age Group')
    ax.set_ylabel('Passenger Count')
    ax.set_xticks(x)
    ax.set_xticklabels(AGE_GROUP_LABELS)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_destination_age_distribution(train_df: pd.DataFrame, output_path: Path) -> None:
    """Destination(목적지)별 연령대 비율 분포를 누적 막대 그래프로 저장한다.

    각 목적지의 전체 승객 수를 1로 정규화하여 비율로 표시하므로
    승객 수 차이에 관계없이 연령 구성을 비교할 수 있다.

    Args:
        train_df:    학습 데이터 (Destination, Age 열 포함).
        output_path: 그래프 이미지 저장 경로.
    """
    # 필요한 열만 선택, 결측 Destination 은 'Unknown' 으로 대체
    work = train_df[['Destination', 'Age']].copy()
    work['Age'] = pd.to_numeric(work['Age'], errors='coerce')
    work['Destination'] = work['Destination'].fillna('Unknown')
    work = work.dropna(subset=['Age'])

    # 나이를 연령대로 변환 후 구간 외 데이터 제거
    work['AgeGroup'] = make_age_group_series(work['Age'])
    work = work.dropna(subset=['AgeGroup'])

    # Destination × AgeGroup 크로스탭 (열 순서를 AGE_GROUP_LABELS 로 고정)
    count_table = (
        work.groupby(['Destination', 'AgeGroup'], observed=False)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=AGE_GROUP_LABELS, fill_value=0)
    )

    # 행 합계로 나누어 비율로 변환 (합계가 0인 행은 NaN → 0.0 으로 처리)
    rate_table = count_table.div(
        count_table.sum(axis=1).replace(0, np.nan),
        axis=0,
    ).fillna(0.0)

    fig, ax = plt.subplots(figsize=(11, 6))
    bottom = np.zeros(len(rate_table))  # 누적 막대의 기준선

    for label in AGE_GROUP_LABELS:
        values = rate_table[label].to_numpy()
        ax.bar(rate_table.index, values, bottom=bottom, label=label)
        bottom += values

    ax.set_title('Age Group Ratio by Destination')
    ax.set_xlabel('Destination')
    ax.set_ylabel('Ratio')
    ax.set_ylim(0, 1)
    ax.legend(title='Age Group', bbox_to_anchor=(1.02, 1), loc='upper left')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    """과제 전체 흐름을 순서대로 실행하는 진입점 함수.

    실행 순서:
        1. 데이터 읽기 및 병합
        2. 수량 출력
        3. Transported 관련 피처 탐색
        4. 연령대별 Transported 시각화 저장
        5. (보너스) Destination별 연령대 분포 시각화 저장
    """
    # 스크립트가 위치한 폴더를 기준 경로로 사용
    base_path = Path(__file__).resolve().parent

    # 1. 데이터 읽기 및 병합
    train_df, test_df = read_datasets(base_path)
    merged_df = merge_datasets(train_df, test_df)

    # 2. 전체 데이터 수량 출력
    print(f'train 데이터 수: {len(train_df)}')
    print(f'test 데이터 수: {len(test_df)}')
    print(f'병합 데이터 수: {len(merged_df)}')

    # 3. Transported 와 상관관계가 가장 높은 피처 출력
    top_feature, score = find_most_related_feature(train_df)
    print(f'Transported와 가장 관련성이 높은 항목: {top_feature} (절대 상관계수: {score:.4f})')

    # 4. 연령대별 Transported 묶음 막대 그래프 저장
    age_plot_path = base_path / 'plot_age_transported.png'
    plot_transported_by_age_group(train_df, age_plot_path)
    print(f'저장 완료: {age_plot_path.name}')

    # 5. (보너스) Destination별 연령대 비율 누적 막대 그래프 저장
    destination_plot_path = base_path / 'plot_destination_age_distribution.png'
    plot_destination_age_distribution(train_df, destination_plot_path)
    print(f'저장 완료: {destination_plot_path.name}')


if __name__ == '__main__':
    main()
