# run_all.py
"""
按顺序运行 confrontation3 全流水线。
step2 (GEE 抓取) 默认对 real cohort 生效; 如果不需要真实卫星, 可传 --skip-gee.
"""
import argparse
import os
import subprocess
import sys

import config

STEPS = [
    ("step1", "step1_load_structured.py"),
    ("step2", "step2_pull_real_satellite(corrected).py"),
    ("step3", "step3_facility_trends.py"),
    ("step4", "step4_company_trends.py"),
    ("step5", "step5_detect_greenwashing.py"),
    ("step6", "step6_write_submissions.py"),
]


# 需要在启动时回显的 config 变量分组
CONFIG_GROUPS = [
    ("输入根目录", [
        "PACKAGE_DIR",
        "VIRTUAL_DIR", "VIRTUAL_PROXY_FILE", "VIRTUAL_METRICS_FILE",
        "VIRTUAL_GREEN_FILE", "VIRTUAL_REPORTS_DIR",
        "REAL_DIR", "REAL_CONTEXT_FILE", "REAL_METRICS_FILE",
        "REAL_GREEN_FILE", "REAL_REPORTS_DIR",
        "TEMPLATE_DIR", "TEMPLATE_SATELLITE",
        "TEMPLATE_STRUCTURED", "TEMPLATE_TEXT",
    ]),
    ("输出目录", [
        "BASE_DIR", "DATA_DIR",
        "UNIFIED_PROXY_OUTPUT", "UNIFIED_METRICS_OUTPUT",
        "REAL_GEE_OUTPUT",
        "FACILITY_TRENDS_OUTPUT", "COMPANY_TRENDS_OUTPUT", "MERGED_OUTPUT",
        "FINDINGS_OUTPUT",
        "SUBMISSION_SATELLITE", "SUBMISSION_STRUCTURED", "SUBMISSION_TEXT",
    ]),
    ("Cohort 标签", ["COHORT_VIRTUAL", "COHORT_REAL"]),
]


def _fmt_value(v):
    """尝试规范化路径,便于人眼比对。"""
    if isinstance(v, str) and (os.sep in v or "/" in v):
        try:
            return os.path.normpath(v)
        except Exception:
            return v
    return v


def print_config_banner(args):
    """在流水线开跑前把关键参数打印一份。"""
    print("=" * 70)
    print("  confrontation3 pipeline —— 运行参数总览")
    print("=" * 70)

    # 命令行参数
    print("\n[命令行参数]")
    for k, v in sorted(vars(args).items()):
        print(f"  --{k.replace('_', '-')} = {v}")

    # config.py 分组回显
    for title, names in CONFIG_GROUPS:
        print(f"\n[{title}]")
        for name in names:
            if hasattr(config, name):
                print(f"  {name:32s} = {_fmt_value(getattr(config, name))}")

    # dict 类配置单独展开
    if hasattr(config, "THRESHOLDS"):
        print("\n[检测阈值 THRESHOLDS]")
        for k, v in config.THRESHOLDS.items():
            print(f"  {k:32s} = {v}")

    if hasattr(config, "CONFIDENCE_BASE"):
        print("\n[置信度基线 CONFIDENCE_BASE]")
        for k, v in config.CONFIDENCE_BASE.items():
            print(f"  {k:40s} = {v}")
        if hasattr(config, "DEFAULT_CONFIDENCE"):
            print(f"  {'DEFAULT_CONFIDENCE':40s} = {config.DEFAULT_CONFIDENCE}")

    if hasattr(config, "GEE_CONFIG"):
        print("\n[GEE 抓取配置]")
        for k, v in config.GEE_CONFIG.items():
            print(f"  {k:32s} = {_fmt_value(v)}")

    # 待执行步骤
    print("\n[执行步骤]")
    for name, script in STEPS:
        state = "跳过" if (args.skip_gee and name == "step2") else "运行"
        print(f"  [{state}] {name:6s} -> {script}")

    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-gee", action="store_true",
                        help="跳过 step2 (适用于无 GEE 授权/网络的场景)")
    args = parser.parse_args()

    print_config_banner(args)

    for name, script in STEPS:
        if args.skip_gee and name == "step2":
            print(f"⏭  跳过 {name}")
            continue
        print(f"\n===== 运行 {name} ({script}) =====")
        ret = subprocess.run([sys.executable, script])
        if ret.returncode != 0:
            print(f"❌ {name} 失败, 退出")
            sys.exit(ret.returncode)


if __name__ == "__main__":
    main()
