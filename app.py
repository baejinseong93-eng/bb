import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(page_title="FFCell 결함 탐지", layout="wide")

COLORS = {"Normal": "#2a78d6", "NoNose": "#eb6834",
          "NoNose,NoBody2": "#eda100", "NoNose,NoBody2,NoBody1": "#e34948"}
SHORT = {"Normal": "정상", "NoNose": "노즈 없음",
         "NoNose,NoBody2": "노즈+몸통2 없음", "NoNose,NoBody2,NoBody1": "다빠짐"}


@st.cache_data
def load():
    df = pd.read_csv("cycle_features.csv")
    if "cycle_id" not in df.columns:
        df = df.rename(columns={df.columns[0]: "cycle_id"})
    df["cycle_id"] = df["cycle_id"].astype(str)
    p = df["cycle_id"].str.split("_")
    df["session"] = p.str[0]
    df["cycle_no"] = pd.to_numeric(p.str[1], errors="coerce")
    df["is_defect"] = df["label"] != "Normal"
    df["label_kr"] = df["label"].map(SHORT).fillna(df["label"])
    return df.copy()


@st.cache_data
def load_trace():
    try:
        return pd.read_csv("sensor_trace.csv")
    except FileNotFoundError:
        return None


df = load()
trace = load_trace()
ORDER = [c for c in COLORS if c in set(df["label"])]
RANGE = [COLORS[c] for c in ORDER]


def pick(*n):
    for x in n:
        if x in df.columns:
            return x
    return None


NOSE = pick("r03_nose_peak")
GRAB = pick("r04_grab_count")
DUR = pick("cycle_duration", "duration_s")

st.sidebar.title("FFCell 대시보드")
PAGE = st.sidebar.radio("이동", ["종합 현황", "센서 상세", "탐지 근거", "데이터 한계"])
st.sidebar.divider()
sess = st.sidebar.multiselect("세션", sorted(df["session"].unique()),
                              default=sorted(df["session"].unique()))
d = df[df["session"].isin(sess)]
if len(d) == 0:
    st.warning("조건에 맞는 사이클이 없습니다.")
    st.stop()

# ══════════════════════════════════════════════════════════
#  1. 종합 현황
# ══════════════════════════════════════════════════════════
if PAGE == "종합 현황":
    st.title("FFCell — 로봇 조립 결함 탐지 현황")
    st.caption("2023-12-11 13:04 ~ 12-12 18:07 · 로봇 4대(R01~R04) + 컨베이어")

    st.subheader("조립 실적")
    total, ok, ng = len(d), (~d["is_defect"]).sum(), d["is_defect"].sum()
    k = st.columns(4)
    k[0].metric("조립 시도", f"{total:,} 개")
    k[1].metric("정상 완성", f"{ok:,} 개")
    k[2].metric("결함 발생", f"{ng:,} 개", delta=f"-{ng}", delta_color="inverse")
    k[3].metric("수율", f"{ok / total * 100:.1f} %")

    st.divider()
    st.subheader("결함 유형별 발생 건수")
    c1, c2 = st.columns([1.5, 1])
    with c1:
        cnt = d["label"].value_counts().rename_axis("label").reset_index(name="건수")
        cnt["유형"] = cnt["label"].map(SHORT).fillna(cnt["label"])
        st.altair_chart(
            alt.Chart(cnt).mark_bar(cornerRadiusEnd=4, height=42).encode(
                x=alt.X("건수:Q", title="사이클 수"),
                y=alt.Y("유형:N", title=None, sort="-x"),
                color=alt.Color("label:N", scale=alt.Scale(domain=ORDER, range=RANGE),
                                legend=None),
                tooltip=["유형:N", "건수:Q"],
            ).properties(height=230), width="stretch")
    with c2:
        t = cnt[["유형", "건수"]].copy()
        t["비율"] = (t["건수"] / total * 100).round(1).astype(str) + "%"
        st.dataframe(t, width="stretch", hide_index=True)

    st.divider()
    st.subheader("탐지 성능")
    p = st.columns(3)
    p[0].metric("정확도", "98.0 %", help="세션 내부 StratifiedKFold 기준")
    p[1].metric("탐지 균형 (macro recall)", "96.0 %")
    p[2].metric("결함 탐지 F1", "98.9 %", help="규칙기반 계층 분류기")

    st.caption("주력 모델: 규칙기반 계층 분류기 · 머신러닝은 경계 사례 보조")

    st.divider()
    st.subheader("클래스별 탐지율 (recall)")
    rec = pd.DataFrame({
        "클래스": [SHORT[c] for c in ORDER],
        "탐지율": [0.99, 0.88, 0.98, 1.00][:len(ORDER)],
        "건수": [int(cnt[cnt["label"] == c]["건수"].iloc[0]) if c in set(cnt["label"]) else 0
                 for c in ORDER],
    })
    st.altair_chart(
        alt.Chart(rec).mark_bar(cornerRadiusEnd=4, height=38).encode(
            x=alt.X("탐지율:Q", scale=alt.Scale(domain=[0, 1]), title="탐지율"),
            y=alt.Y("클래스:N", title=None, sort=[SHORT[c] for c in ORDER]),
            color=alt.Color("클래스:N", legend=None,
                            scale=alt.Scale(domain=[SHORT[c] for c in ORDER], range=RANGE)),
            tooltip=["클래스:N", "탐지율:Q", "건수:Q"],
        ).properties(height=200), width="stretch")

    st.info("노즈 없음(88%)이 가장 낮습니다 — 가장 작은 부품이라 로봇 동작에 티가 "
            "잘 나지 않기 때문입니다. 자세한 근거는 **탐지 근거** 페이지를 참고하세요.")

# ══════════════════════════════════════════════════════════
#  2. 센서 상세
# ══════════════════════════════════════════════════════════
elif PAGE == "센서 상세":
    st.title("센서 상세 — 로봇은 실제로 어떻게 움직이는가")

    if trace is None:
        st.error("`sensor_trace.csv`가 없습니다. 노트북에서 생성 후 같은 폴더에 두세요.")
        st.stop()

    st.caption("사이클 1회(약 295초) 동안 각 로봇의 그리퍼 신호 변화")

    sensors = [c for c in trace.columns if c.startswith("I_")]
    pick_s = st.selectbox("센서 선택", sensors,
                          index=sensors.index("I_R03_Gripper_Load")
                          if "I_R03_Gripper_Load" in sensors else 0)
    pick_l = st.multiselect("비교할 클래스", list(SHORT.values()),
                            default=["정상", "노즈 없음"])

    tr = trace.copy()
    tr["클래스"] = tr["label"].map(SHORT)
    tr = tr[tr["클래스"].isin(pick_l)]

    st.altair_chart(
        alt.Chart(tr).mark_line(opacity=0.85, strokeWidth=1.5).encode(
            x=alt.X("t:Q", title="사이클 경과 시간 (초)"),
            y=alt.Y(f"{pick_s}:Q", title=pick_s),
            color=alt.Color("클래스:N",
                            scale=alt.Scale(domain=list(SHORT.values()),
                                            range=list(COLORS.values())),
                            legend=alt.Legend(orient="bottom", title=None)),
            tooltip=["클래스:N", "t:Q", f"{pick_s}:Q"],
        ).properties(height=340), width="stretch")

    if "R03" in pick_s and "Load" in pick_s:
        st.success("**R03은 부품을 눌러 조립하는 로봇입니다.** 부품마다 힘 봉우리가 생깁니다 — "
                   "약 66초 Body1, 79초 Body2, **97초 노즈**, 103초 Tail. "
                   "부품이 없으면 그 자리 봉우리만 사라집니다.")
    elif "R04" in pick_s:
        st.info("**R04는 분해 로봇입니다.** 192~230초 구간이 분해, 230~300초가 복귀입니다. "
                "노즈가 없어도 같은 깊이로 그리퍼를 닫기 때문에 Pot만으로는 구분되지 않습니다.")

    st.divider()
    st.subheader("전 로봇 한눈에 보기")
    pick_one = st.selectbox("클래스", list(SHORT.values()), index=0)
    one = trace[trace["label"].map(SHORT) == pick_one]
    long = one.melt(id_vars=["t"], value_vars=sensors,
                    var_name="센서", value_name="값")
    st.altair_chart(
        alt.Chart(long).mark_line(strokeWidth=1.2).encode(
            x=alt.X("t:Q", title="시간 (초)"),
            y=alt.Y("값:Q", title=None),
            color=alt.Color("센서:N", legend=None),
            row=alt.Row("센서:N", title=None,
                        header=alt.Header(labelAngle=0, labelAlign="left")),
        ).properties(height=80, width=760), width="content")

# ══════════════════════════════════════════════════════════
#  3. 탐지 근거
# ══════════════════════════════════════════════════════════
elif PAGE == "탐지 근거":
    st.title("탐지 근거 — 무엇을 보고 판별하는가")

    if GRAB:
        st.subheader("1. 부품 개수 — R04 딥그랩 횟수")
        st.caption("분해 1회 + 되돌리기 1회 = 부품당 2회. 부품이 빠질수록 2회씩 줄어듭니다.")
        g = d.groupby(["label_kr", d[GRAB].astype(int)]).size().reset_index(name="건수")
        g.columns = ["클래스", "딥그랩 횟수", "건수"]
        st.altair_chart(
            alt.Chart(g).mark_circle(opacity=0.85).encode(
                x=alt.X("딥그랩 횟수:O"),
                y=alt.Y("클래스:N", title=None, sort=[SHORT[c] for c in ORDER]),
                size=alt.Size("건수:Q", scale=alt.Scale(range=[100, 1500]),
                              legend=alt.Legend(title="사이클 수")),
                color=alt.Color("클래스:N", legend=None,
                                scale=alt.Scale(domain=[SHORT[c] for c in ORDER],
                                                range=RANGE)),
                tooltip=["클래스:N", "딥그랩 횟수:O", "건수:Q"],
            ).properties(height=240), width="stretch")
        st.warning("몸통2 없음(7회)·다빠짐(9회)은 이것만으로 갈립니다. "
                   "하지만 **정상과 노즈 없음은 둘 다 5회**라 구분되지 않습니다.")

    st.divider()
    st.subheader("2. 노즈 유무 — R03 97초 구간 조립 힘")

    if NOSE:
        nz = d[d["label"].isin(["Normal", "NoNose"])].copy()
        nz[NOSE] = pd.to_numeric(nz[NOSE], errors="coerce")
        nz = nz.dropna(subset=[NOSE])
        c1, c2 = st.columns([1.4, 1])
        with c1:
            st.altair_chart(
                alt.Chart(nz).mark_circle(size=70, opacity=0.65).encode(
                    x=alt.X(f"{NOSE}:Q", title="97초 구간 최대 힘",
                            scale=alt.Scale(zero=False)),
                    y=alt.Y("label_kr:N", title=None),
                    color=alt.Color("label:N", legend=None,
                                    scale=alt.Scale(domain=["Normal", "NoNose"],
                                                    range=[COLORS["Normal"],
                                                           COLORS["NoNose"]])),
                    tooltip=["cycle_id:N", "label_kr:N", f"{NOSE}:Q"],
                ).properties(height=200), width="stretch")
        with c2:
            st.dataframe(nz.groupby("label_kr")[NOSE].agg(["mean", "min", "max"]).round(0),
                         width="stretch")
        st.success("**완전히 갈립니다.** 노즈가 있으면 97초에 힘이 솟고(약 6564), "
                   "없으면 평소값 그대로입니다(약 1399).")
    else:
        st.markdown("""
| 측정 범위 | 정상 | 노즈 없음 | 배율 |
|---|---|---|---|
| 사이클 전체 평균 | 1,851 | 1,803 | 1.03배 |
| **97초 구간 최대값** | **6,564** | **1,399** | **4.7배** |
        """)
        st.info("`r03_nose_peak` 컬럼을 CSV에 포함하면 이 자리에 분포 그래프가 나타납니다.")

    st.divider()
    st.subheader("3. 피처 중요도 — 시간 오염 검증")
    st.caption("라벨을 고정하고 잰 시간 상관. 0.5 이상이면 결함이 아니라 '수집 시점'을 학습할 위험")
    contam = pd.DataFrame({
        "피처": ["Q_Cell_CycleCount", "I_R04_Gripper_Load", "I_R01_Gripper_Load/Pot",
                 "I_R02_Gripper_Load_mean", "Q_VFD 온도", "r04_Body1/Body2_제거",
                 "r03_nose_peak", "r03_nose_rise_slope", "r03_nose_fwhm"],
        "시간상관": [0.952, 0.775, 0.640, 0.630, 0.590, 0.570, 0.234, 0.111, 0.077],
    })
    contam["판정"] = contam["시간상관"].apply(lambda v: "오염 — 제거" if v >= 0.5 else "무죄 — 유지")
    st.altair_chart(
        alt.Chart(contam).mark_bar(cornerRadiusEnd=3).encode(
            x=alt.X("시간상관:Q", title="라벨통제 시간 상관"),
            y=alt.Y("피처:N", title=None, sort="-x"),
            color=alt.Color("판정:N",
                            scale=alt.Scale(domain=["오염 — 제거", "무죄 — 유지"],
                                            range=["#e34948", "#1baf7a"]),
                            legend=alt.Legend(orient="bottom", title=None)),
            tooltip=["피처:N", "시간상관:Q", "판정:N"],
        ).properties(height=300), width="stretch")
    st.success("**R03 노즈 신호는 0.08~0.23으로 무죄입니다.** 진짜 결함 신호라는 근거입니다.")

    with st.expander("오염 제거 효과 자세히 보기"):
        st.dataframe(pd.DataFrame({
            "조건": ["VFD·R01 유지", "VFD만 제거", "R01만 제거", "둘 다 제거"],
            "피처 수": [100, 86, 93, 79],
            "정답률": [0.000, 0.000, 0.000, 0.913],
        }), width="stretch", hide_index=True)
        st.caption("오염 경로가 하나라도 남으면 판별이 무너집니다. "
                   "시드 10회 반복 시 최종 피처셋은 0.913 ± 0.000으로 완전히 안정적입니다.")

# ══════════════════════════════════════════════════════════
#  4. 데이터 한계
# ══════════════════════════════════════════════════════════
elif PAGE == "데이터 한계":
    st.title("데이터 한계 — 정직하게 남기는 기록")

    st.subheader("1. 라벨이 시간 블록으로 뭉쳐 있다")
    blocks = pd.DataFrame({
        "시작": ["12/11 13:04", "12/11 15:36", "12/11 19:59",
                 "12/12 00:16", "12/12 04:45", "12/12 16:14"],
        "종료": ["12/11 15:31", "12/11 19:55", "12/12 00:11",
                 "12/12 04:40", "12/12 16:09", "12/12 18:02"],
        "라벨": ["Normal", "NoNose", "NoNose,NoBody2",
                 "NoNose,NoBody2,NoBody1", "Normal", "NoNose"],
        "개수": [23, 36, 51, 54, 138, 23],
    })
    blocks["순서"] = range(len(blocks))
    blocks["유형"] = blocks["라벨"].map(SHORT)
    st.altair_chart(
        alt.Chart(blocks).mark_bar(cornerRadius=2, size=26).encode(
            x=alt.X("개수:Q", stack="zero", title="사이클 수 (수집 순서)",
                    scale=alt.Scale(domain=[0, 325], nice=False),
                    axis=alt.Axis(values=[0, 50, 100, 150, 200, 250, 300, 325],
                                  labelPadding=6, tickSize=4)),
            color=alt.Color("유형:N",
                            scale=alt.Scale(domain=[SHORT[c] for c in COLORS],
                                            range=list(COLORS.values())),
                            legend=None),
            order=alt.Order("순서:Q"),
            tooltip=["시작:N", "종료:N", "유형:N", "개수:Q"],
        ).properties(height=70), width="stretch")
    st.caption("파랑=정상 · 주황=노즈 없음 · 노랑=노즈+몸통2 없음 · 빨강=다빠짐")    
    st.dataframe(blocks[["시작", "종료", "유형", "개수"]], width="stretch", hide_index=True)
    st.warning("한 종류를 몰아서 수집했기 때문에, 센서가 시간에 따라 조금만 변해도 "
               "그 변화가 라벨과 상관을 갖게 됩니다. 오염 피처를 걸러낸 이유입니다.")

    st.divider()
    st.subheader("2. 세션 1~4에는 2개 클래스가 아예 없다")
    st.dataframe(pd.DataFrame({
        "클래스": [SHORT[c] for c in ["Normal", "NoNose",
                                     "NoNose,NoBody2", "NoNose,NoBody2,NoBody1"]],
        "세션 1~4": [18, 27, 0, 0],
        "세션 5": [137, 32, 51, 54],
    }), width="stretch", hide_index=True)
    st.error("GroupKFold가 세션5를 시험으로 빼면 학습 데이터에 그 클래스가 한 번도 "
             "등장하지 않습니다. 배운 적 없는 것을 맞히라는 구조라 어떤 모델을 써도 "
             "실패합니다 — 피처가 아니라 데이터 구조의 한계입니다.")

    st.divider()
    st.subheader("3. 라벨 오류로 확정된 2건")
    st.dataframe(pd.DataFrame({
        "사이클": ["5_253", "5_265"],
        "라벨": ["NoNose", "NoNose"],
        "R03 97초 힘": [6775, 6742],
        "해석": ["봉우리 있음 → 노즈 있었음", "봉우리 있음 → 노즈 있었음"],
    }), width="stretch", hide_index=True)
    st.caption("나머지 21개는 1391~1407로 노즈 없음이 맞습니다. "
               "독립된 두 갈래 분석이 동일하게 이 2건을 지목했습니다.")

    st.divider()
    st.subheader("4. 남은 과제")
    st.markdown("- 세션 1~4 클래스 결측 — 추가 수집 또는 평가 방식 재설계\n"
                "- 이미지 CV 데이터 확보 후 센서와 교차 검증\n"
                "- n8n 기반 품질 리포팅 자동화\n"
                "- 5_253 · 5_265 영상 로그 대조로 라벨 최종 확정")
