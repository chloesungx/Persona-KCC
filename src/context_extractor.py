"""Optional live-news adapter.

The publication experiments use the bundled, frozen article pool. Live crawling was
part of data preparation and is intentionally excluded from this code snapshot.
"""


async def fetch_naver_news_summary(*args, **kwargs):
    del args, kwargs
    raise RuntimeError(
        "Live crawling is not included. Use data/external/data_VCF0842_sample_balance.json."
    )
