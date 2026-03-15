"""
setup.py for cli-anything-trading-bot

Install with: pip install -e .
"""

from setuptools import setup, find_namespace_packages

setup(
    name="cli-anything-trading-bot",
    version="1.0.0",
    author="cli-anything contributors",
    description="CLI harness for Kalshi BTC Range trading bot — agent-native interface",
    url="https://github.com/caseybattle/Trading-agent",
    packages=find_namespace_packages(include=["cli_anything.*"]),
    python_requires=">=3.10",
    install_requires=[
        "click>=8.0.0",
        "prompt-toolkit>=3.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "trading-bot=cli_anything.trading_bot.trading_bot_cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
