# -*- coding: utf-8 -*-
"""
T90 骑射预瞄提前量计算器 - Streamlit 网页版

运行：
    streamlit run app.py

说明：
- 简化模型，不计算风、空气阻力、撒放延迟、马加速度、拉距变化。
- 骑手位置直接等同为马的位置。
- 默认靶心纵向位置为计时跑道中点 45 m。
- 默认横向距离为 10 m，即 T90 规则中的 9 m + 马在跑道中央约 1 m。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple, List

import pandas as pd
import streamlit as st


G = 9.81

TIMED_LANE_LENGTH_M = 90.0
TARGET_X_M = TIMED_LANE_LENGTH_M / 2.0
DEFAULT_LATERAL_DISTANCE_M = 10.0
DEFAULT_TARGET_DIAMETER_M = 0.90
DEFAULT_TARGET_HEIGHT_M = 1.80

LBF_TO_N = 4.4482216153
DEFAULT_POWER_STROKE_M = 0.5334
DEFAULT_BOW_EFFICIENCY = 0.68


@dataclass(frozen=True)
class TargetType:
    name: str
    angle_deg: float
    recommended_range: Tuple[float, float]
    note: str


TARGETS = {
    "分鬃靶": TargetType(
        name="分鬃靶",
        angle_deg=114.0,
        recommended_range=(10.0, 35.0),
        note="侧向起点，靶面平面相对跑道方向约 114°。",
    ),
    "对镫靶": TargetType(
        name="对镫靶",
        angle_deg=0.0,
        recommended_range=(30.0, 70.0),
        note="平行赛道，靶面平面角度按 0° 处理。",
    ),
    "抹楸靶": TargetType(
        name="抹楸靶",
        angle_deg=66.0,
        recommended_range=(65.0, 90.0),
        note="侧向终点；程序用分鬃镜像角 66° 计算。",
    ),
}


def estimate_arrow_speed_from_bow(draw_weight_lb: float, arrow_weight_g: float) -> tuple[float, float, float]:
    """用弓磅数和箭重估算箭速。返回 default, low, high。"""
    force_n = draw_weight_lb * LBF_TO_N
    arrow_mass_kg = arrow_weight_g / 1000.0

    stored_energy_j = 0.5 * force_n * DEFAULT_POWER_STROKE_M
    speed_default = math.sqrt(2.0 * stored_energy_j * DEFAULT_BOW_EFFICIENCY / arrow_mass_kg)
    speed_low = math.sqrt(2.0 * stored_energy_j * 0.60 / arrow_mass_kg)
    speed_high = math.sqrt(2.0 * stored_energy_j * 0.75 / arrow_mass_kg)

    return speed_default, speed_low, speed_high


def estimate_arrow_speed_from_calibration(df: pd.DataFrame, target_diameter_m: float) -> tuple[float, pd.DataFrame]:
    """
    用步射距离 + 上瞄/落低靶径倍数估算有效箭速。
    drop = aim_diameters * target_diameter
    t = sqrt(2*drop/g)
    speed = distance/t
    """
    rows = []
    for _, row in df.iterrows():
        try:
            distance_m = float(row["步射距离 m"])
            aim_diameters = float(row["上瞄/落低量（靶径倍数）"])
        except Exception:
            continue

        if distance_m <= 0 or aim_diameters <= 0:
            continue

        drop_m = aim_diameters * target_diameter_m
        t = math.sqrt(2.0 * drop_m / G)
        speed = distance_m / t

        rows.append({
            "步射距离 m": distance_m,
            "上瞄/落低量（靶径倍数）": aim_diameters,
            "等效下坠 m": drop_m,
            "估算有效箭速 m/s": speed,
        })

    result_df = pd.DataFrame(rows)
    if result_df.empty:
        raise ValueError("请至少输入一组有效的步射校准数据。距离和靶径倍数都必须大于 0。")

    return float(result_df["估算有效箭速 m/s"].mean()), result_df


def solve_flight_time_2d(d_x: float, lateral_m: float, horse_speed: float, arrow_speed: float) -> float:
    """
    解方程：
        (D - U*t)^2 + L^2 = (V*t)^2

    D = 靶心相对骑手的前后距离
    U = 马速
    L = 横向距离
    V = 箭相对骑手/弓的箭速
    """
    if arrow_speed <= horse_speed:
        raise ValueError("箭速必须大于马速，否则简化模型无法计算。")

    a = arrow_speed ** 2 - horse_speed ** 2
    b = 2.0 * d_x * horse_speed
    c = -(d_x ** 2 + lateral_m ** 2)

    disc = b ** 2 - 4.0 * a * c
    if disc < 0:
        raise ValueError("方程无实数解，请检查输入。")

    t1 = (-b + math.sqrt(disc)) / (2.0 * a)
    t2 = (-b - math.sqrt(disc)) / (2.0 * a)
    candidates = [t for t in (t1, t2) if t > 0]
    if not candidates:
        raise ValueError("没有正的飞行时间解，请检查输入。")

    return min(candidates)


def target_face_offset(
    rider_x_m: float,
    target_x_m: float,
    lateral_m: float,
    lead_along_track_m: float,
    target_angle_deg: float,
) -> float:
    """
    计算“虚拟瞄点”投影/交会到靶面平面后，距离靶心的靶面等效偏移量。

    S = 骑手/马位置
    C = 靶心
    A = 靶心向起点侧移动 lead 后的虚拟瞄点
    射线 S->A 与靶面线 C + r*e 的交点，|r| 就是靶面等效偏移。
    """
    sx, sy = rider_x_m, 0.0
    cx, cy = target_x_m, lateral_m
    ax, ay = target_x_m - lead_along_track_m, lateral_m

    vx, vy = ax - sx, ay - sy

    theta = math.radians(target_angle_deg)
    ex, ey = math.cos(theta), math.sin(theta)

    # 解：S + q*v = C + r*e
    # q*v - r*e = C-S
    a, b = vx, -ex
    c, d = vy, -ey
    e, f = cx - sx, cy - sy

    det = a * d - b * c

    if abs(det) < 1e-9:
        return abs(lead_along_track_m * math.cos(theta))

    r = (a * f - e * c) / det
    return abs(r)


def calculate_result(
    target: TargetType,
    rider_x_m: float,
    horse_speed: float,
    arrow_speed: float,
    lateral_m: float,
    target_diameter_m: float,
) -> dict:
    d_x = TARGET_X_M - rider_x_m

    t = solve_flight_time_2d(
        d_x=d_x,
        lateral_m=lateral_m,
        horse_speed=horse_speed,
        arrow_speed=arrow_speed,
    )

    lead_along = horse_speed * t
    face_offset = target_face_offset(
        rider_x_m=rider_x_m,
        target_x_m=TARGET_X_M,
        lateral_m=lateral_m,
        lead_along_track_m=lead_along,
        target_angle_deg=target.angle_deg,
    )
    drop_m = 0.5 * G * t ** 2

    return {
        "靶心相对骑手前后距离 D m": d_x,
        "水平直线距离 R m": math.sqrt(d_x ** 2 + lateral_m ** 2),
        "飞行时间 s": t,
        "沿跑道等效提前量 m": lead_along,
        "靶面等效偏移 m": face_offset,
        "从靶心向起点侧 靶径": face_offset / target_diameter_m,
        "超过靶边 靶径": max(0.0, face_offset - target_diameter_m / 2.0) / target_diameter_m,
        "重力下坠补偿 m": drop_m,
        "向上补偿 靶径": drop_m / target_diameter_m,
    }


def fmt(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}"


st.set_page_config(
    page_title="T90 骑射预瞄计算器",
    page_icon="🏹",
    layout="wide",
)

st.title("🏹 T90 骑射预瞄提前量计算器")
st.caption("简化模型：不计算风、空气阻力、撒放延迟、马加速度、拉距变化。输出用于训练估算，不替代实地校准。")

with st.sidebar:
    st.header("1. 靶型与位置")

    target_name = st.selectbox("靶型", list(TARGETS.keys()), index=1)
    target = TARGETS[target_name]
    st.info(target.note)

    rec_low, rec_high = target.recommended_range
    rider_x_m = st.number_input(
        "骑手/马在计时跑道上的位置 m",
        min_value=0.0,
        max_value=90.0,
        value=float((rec_low + rec_high) / 2.0),
        step=1.0,
    )

    if not (rec_low <= rider_x_m <= rec_high):
        st.warning(f"当前不在 {target.name} 推荐射击区间 {rec_low:.0f}–{rec_high:.0f} m 内。")

    st.header("2. 马速")
    horse_mode = st.radio(
        "马速输入方式",
        ["输入距离和用时", "直接输入马速"],
        horizontal=False,
    )

    if horse_mode == "直接输入马速":
        horse_speed = st.number_input("马速 m/s", min_value=0.1, value=7.14, step=0.1)
    else:
        run_distance = st.number_input("跑过的距离 m", min_value=0.1, value=100.0, step=1.0)
        run_time = st.number_input("用时 s", min_value=0.1, value=14.0, step=0.1)
        horse_speed = run_distance / run_time
        st.caption(f"计算得到马速：{horse_speed:.2f} m/s")

    st.header("3. 场地与靶面")
    lateral_m = st.number_input(
        "骑手到靶心横向距离 m",
        min_value=0.1,
        value=DEFAULT_LATERAL_DISTANCE_M,
        step=0.1,
        help="T90 默认可用：跑道内侧到靶心 9 m + 马在跑道中央约 1 m = 10 m。",
    )
    target_diameter_m = st.number_input(
        "靶面直径 m",
        min_value=0.1,
        value=DEFAULT_TARGET_DIAMETER_M,
        step=0.01,
    )
    target_height_m = st.number_input(
        "靶心高度 m",
        min_value=0.1,
        value=DEFAULT_TARGET_HEIGHT_M,
        step=0.05,
        help="当前简化模型主要用它作记录；垂直下坠默认以靶心高度附近发射来估算。",
    )

st.header("4. 箭速来源")
speed_mode = st.radio(
    "选择箭速来源",
    ["直接输入实测箭速", "用弓磅数 + 箭重估算", "用步射上瞄/下坠数据估算有效箭速"],
    index=1,
)

calibration_df = None
arrow_speed_note = ""

if speed_mode == "直接输入实测箭速":
    arrow_speed = st.number_input("实测箭速 m/s", min_value=1.0, value=47.5, step=0.5)
    arrow_speed_note = "使用实测箭速。"

elif speed_mode == "用弓磅数 + 箭重估算":
    col_a, col_b = st.columns(2)
    with col_a:
        draw_weight_lb = st.number_input("弓磅数 lb（按 28 英寸拉距标称）", min_value=1.0, value=30.0, step=1.0)
    with col_b:
        arrow_weight_g = st.number_input("箭重 g", min_value=1.0, value=22.0, step=0.5)

    arrow_speed, speed_low, speed_high = estimate_arrow_speed_from_bow(draw_weight_lb, arrow_weight_g)
    arrow_speed_note = (
        f"简化估算箭速：{arrow_speed:.2f} m/s；粗略范围：{speed_low:.2f}–{speed_high:.2f} m/s。"
        "默认效率 0.68、有效做功距离约 21 英寸。"
    )

else:
    st.write("输入几组步射数据。可以填“为了命中靶心需要上瞄多少靶径”，或“固定瞄靶心时平均落低多少靶径”。")
    default_df = pd.DataFrame(
        {
            "步射距离 m": [10.0, 20.0, 30.0],
            "上瞄/落低量（靶径倍数）": [0.12, 0.45, 0.95],
        }
    )
    user_df = st.data_editor(
        default_df,
        num_rows="dynamic",
        use_container_width=True,
    )

    try:
        arrow_speed, calibration_df = estimate_arrow_speed_from_calibration(user_df, target_diameter_m)
        sd = float(calibration_df["估算有效箭速 m/s"].std()) if len(calibration_df) > 1 else 0.0
        arrow_speed_note = f"根据步射数据估算有效箭速：{arrow_speed:.2f} m/s；组间标准差：{sd:.2f} m/s。"
    except Exception as exc:
        arrow_speed = None
        st.error(str(exc))

if arrow_speed is not None:
    st.success(arrow_speed_note)

    try:
        result = calculate_result(
            target=target,
            rider_x_m=rider_x_m,
            horse_speed=horse_speed,
            arrow_speed=arrow_speed,
            lateral_m=lateral_m,
            target_diameter_m=target_diameter_m,
        )

        st.header("5. 计算结果")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("飞行时间", f"{result['飞行时间 s']:.3f} s")
        col2.metric("沿跑道提前量", f"{result['沿跑道等效提前量 m']:.2f} m")
        col3.metric("靶面等效偏移", f"{result['靶面等效偏移 m']:.2f} m")
        col4.metric("重力上瞄补偿", f"{result['重力下坠补偿 m']:.2f} m")

        st.subheader("建议瞄点")
        st.markdown(
            f"""
            **向起点侧 {result['从靶心向起点侧 靶径']:.2f} 个靶径，向上 {result['向上补偿 靶径']:.2f} 个靶径。**

            也就是：向起点侧约 **{result['靶面等效偏移 m']:.2f} m**，
            向上约 **{result['重力下坠补偿 m']:.2f} m**。
            """
        )

        with st.expander("详细数值"):
            details = pd.DataFrame(
                {
                    "项目": [
                        "靶型",
                        "骑手位置",
                        "靶心纵向位置",
                        "靶心相对骑手前后距离 D",
                        "横向距离 L",
                        "水平直线距离 R",
                        "马速",
                        "箭速",
                        "靶面角度",
                        "靶面直径",
                        "靶心高度",
                        "箭飞行时间",
                        "沿跑道等效提前量",
                        "靶面等效偏移",
                        "从靶心向起点侧",
                        "超过靶边",
                        "重力下坠补偿",
                        "向上补偿",
                    ],
                    "数值": [
                        target.name,
                        f"{rider_x_m:.2f} m",
                        f"{TARGET_X_M:.2f} m",
                        f"{result['靶心相对骑手前后距离 D m']:.2f} m",
                        f"{lateral_m:.2f} m",
                        f"{result['水平直线距离 R m']:.2f} m",
                        f"{horse_speed:.2f} m/s",
                        f"{arrow_speed:.2f} m/s",
                        f"{target.angle_deg:.1f}°",
                        f"{target_diameter_m:.2f} m",
                        f"{target_height_m:.2f} m",
                        f"{result['飞行时间 s']:.3f} s",
                        f"{result['沿跑道等效提前量 m']:.2f} m",
                        f"{result['靶面等效偏移 m']:.2f} m",
                        f"{result['从靶心向起点侧 靶径']:.2f} 个靶径",
                        f"{result['超过靶边 靶径']:.2f} 个靶径",
                        f"{result['重力下坠补偿 m']:.2f} m",
                        f"{result['向上补偿 靶径']:.2f} 个靶径",
                    ],
                }
            )
            st.dataframe(details, use_container_width=True, hide_index=True)

        if calibration_df is not None:
            with st.expander("步射校准数据计算过程"):
                st.dataframe(calibration_df, use_container_width=True, hide_index=True)

        st.caption(
            "说明：这里的“向起点侧”指沿赛道方向朝起点一侧修正。"
            "分鬃靶、抹楸靶因靶面斜放，程序会把沿跑道提前量转换为靶面等效偏移。"
        )

    except Exception as exc:
        st.error(f"计算失败：{exc}")
else:
    st.warning("请先输入有效箭速或有效校准数据。")
