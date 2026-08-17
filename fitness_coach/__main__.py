"""Entry point: `python -m fitness_coach`."""

import asyncio
import logging
import sys

from .bot import FitnessBot
from .coach import AICoach, warm_up_backend
from .config import Config
from .storage import Storage
from .telegram_api import TelegramClient

logger = logging.getLogger("fitness_coach")


async def _run() -> int:
    config = Config.from_env()
    error = config.validate()
    if error:
        logger.error("%s", error)
        return 2

    storage = Storage(config.database_path)
    client = TelegramClient(config.telegram_token)
    coach = AICoach(config)

    if config.llm_enabled:
        if await warm_up_backend(coach):
            logger.info("LLM backend ready: %s", config.llm_model)
        else:
            logger.warning(
                "LLM backend unreachable — the coach will use offline answers."
            )
    else:
        logger.warning(
            "FITNESS_LLM_API_KEY is not set — the coach will use offline answers."
        )

    bot = FitnessBot(config, storage, client, coach)
    try:
        await bot.run()
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Shutting down")
    finally:
        await client.close()
        await coach.close()
        storage.close()
    return 0


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        return asyncio.run(_run())
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
