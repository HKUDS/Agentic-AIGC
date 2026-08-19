"""Entry point: `python -m fitness_coach`.

Runs the Telegram bot, the web sign-up app, or both in one process, depending
on what the environment configures. One process keeps a single SQLite file
authoritative and fits a single Railway service.
"""

import asyncio
import logging
import secrets
import sys

from aiohttp import web

from .bot import FitnessBot
from .coach import AICoach, warm_up_backend
from .config import Config
from .storage import Storage
from .telegram_api import TelegramClient
from .webapp.auth import AuthService
from .webapp.server import WebApp

logger = logging.getLogger("fitness_coach")


def resolve_secret(config: Config) -> str:
    """Session-signing secret, generated per start when unset."""
    if config.secret_key:
        return config.secret_key
    logger.warning(
        "FITNESS_SECRET_KEY is not set — using a random one, so every restart "
        "signs users out. Set it in production."
    )
    return secrets.token_urlsafe(32)


async def _run() -> int:
    config = Config.from_env()
    error = config.validate()
    if error:
        logger.error("%s", error)
        return 2

    storage = Storage(config.database_path)
    storage.purge_expired_auth_codes()

    run_bot = config.bot_enabled and config.telegram_enabled
    if config.bot_enabled and not config.telegram_enabled:
        logger.warning("TELEGRAM_BOT_TOKEN is not set — running the web app only.")

    client = None
    coach = None
    bot = None
    runner = None
    tasks = []

    try:
        if run_bot:
            client = TelegramClient(config.telegram_token)
            coach = AICoach(config)
            if config.llm_enabled:
                if await warm_up_backend(coach):
                    logger.info("LLM backend ready: %s", config.llm_model)
                else:
                    logger.warning("LLM backend unreachable — using offline answers.")
            else:
                logger.warning("No LLM key — the coach will use offline answers.")

            bot = FitnessBot(config, storage, client, coach)
            tasks.append(asyncio.create_task(bot.run(), name="telegram-bot"))

        if config.web_enabled:
            auth = AuthService(storage, resolve_secret(config))
            webapp = WebApp(config, storage, auth=auth)
            runner = web.AppRunner(webapp.app)
            await runner.setup()
            site = web.TCPSite(runner, config.web_host, config.web_port)
            await site.start()
            logger.info(
                "Web app listening on http://%s:%s", config.web_host, config.web_port
            )

        if not tasks and runner is None:
            logger.error("Neither the bot nor the web app is enabled.")
            return 2

        if tasks:
            await asyncio.gather(*tasks)
        else:
            # Web-only: idle until the process is stopped.
            await asyncio.Event().wait()
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Shutting down")
    finally:
        for task in tasks:
            task.cancel()
        for task in tasks:
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        if runner is not None:
            await runner.cleanup()
        if client is not None:
            await client.close()
        if coach is not None:
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
