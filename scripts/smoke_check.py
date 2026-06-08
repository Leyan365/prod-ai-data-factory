"""
Smoke checks for the repaired tutorial slice.
"""

import asyncio
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


async def main() -> None:
    import training_data_bot
    from training_data_bot import TrainingDataBot, UnifiedLoader

    bot = TrainingDataBot()
    before = bot.get_statistics()

    tmpdir = ROOT / ".tmp" / "smoke_check"
    tmpdir.mkdir(parents=True, exist_ok=True)

    sample = tmpdir / "sample.txt"
    sample.write_text("This is a small tutorial smoke test document.", encoding="utf-8")

    loader = UnifiedLoader()
    document = await loader.load_single(sample)
    documents = await bot.load_documents(sample)

    after = bot.get_statistics()

    assert training_data_bot.__version__
    assert document.title == "sample"
    assert len(documents) == 1
    assert before["documents"]["total"] == 0
    assert after["documents"]["total"] == 1

    await bot.cleanup()
    print("smoke ok")


if __name__ == "__main__":
    asyncio.run(main())
