# `security-bulletin-utils` repo structure and readme

---

## Overview
This is our group's existing repo for interfacing with security bulletin data.

Note it is the existing refactor and related to this project's Claude Project Knowledge text content with the title "Navigating Existing vs. New Infrastructure: Project Context"

https://gitlab.com/fproj/data-team/security-bulletins-collectors/security-bulletins-utils


## security-bulletins-utils Repo Structure

"""
.
├── .gitignore
├── .gitlab-ci.yml
├── CHANGELOG.md
├── README.md
├── poetry.lock
├── pyproject.toml
├── repo_structure.txt
├── security_bulletins_utils
│  ├── __init__.py
│  ├── bigquery
│  │  ├── __init__.py
│  │  ├── bigquery.py
│  │  ├── models.py
│  │  └── schemas.py
│  ├── crawler_release_version.py
│  ├── getter
│  │  ├── __init__.py
│  │  ├── advisory_process_builder.py
│  │  └── base.py
│  ├── git
│  │  ├── __init__.py
│  │  ├── manager.py
│  │  └── models.py
│  ├── llm
│  │  ├── __init__.py
│  │  ├── example.py
│  │  ├── llm.py
│  │  ├── models.py
│  │  └── parser.py
│  ├── models.py
│  ├── mongodb
│  │  ├── __init__.py
│  │  ├── example_package_model
│  │  │  ├── __init__.py
│  │  │  ├── example.py
│  │  │  └── main.py
│  │  ├── example_push_advisories_packages
│  │  │  ├── __init__.py
│  │  │  └── example.py
│  │  ├── models.py
│  │  └── mongodb.py
│  ├── parser
│  │  ├── __init__.py
│  │  ├── builder.py
│  │  ├── example
│  │  │  ├── __init__.py
│  │  │  ├── builder.py
│  │  │  ├── main.py
│  │  │  ├── models.py
│  │  │  ├── parser.py
│  │  │  └── settings.py
│  │  ├── identifier.py
│  │  ├── models.py
│  │  ├── parser.py
│  │  └── settings.py
│  ├── storage
│  │  ├── __init__.py
│  │  └── gcs.py
│  ├── utils.py
│  └── vls
│      ├── __init__.py
│      ├── builder.py
│      ├── example.py
│      ├── exceptions.py
│      ├── merger.py
│      ├── models.py
│      └── schemas.py
└── tests
    ├── __init__.py
    ├── getter
    │  ├── __init__.py
    │  └── test_base.py
    ├── git
    │  └── test_manager.py
    └── vls
        ├── __init__.py
        ├── test_merger.py
        └── test_vls.py



"""

# security-bulletins-utils README.md:

```
# Security Bulletins Utils
![coverage report](https://gitlab.com/fproj/data-team/security-bulletins-collectors/security-bulletins-utils/badges/main/coverage.svg)

This library contains an interface to scrap web pages and store the data (mainly .html files) in GCS and metadata in BigQuery.

## Installation

Pip install the package:

```bashs
pip install security-bulletins-utils --index-url https://__token__:<your_personal_token>@gitlab.com/api/v4/projects/62421193/packages/pypi/simple
```

Or, if you have the GitLab access token configured in the `.pypirc` file, then run:

```bashs
pip install security-bulletins-utils --index-url https://gitlab.com/api/v4/projects/62421193/packages/pypi/simple
```
Ensure that you authenticate with Google Cloud using valid credentials.
This can be done by setting the `GOOGLE_APPLICATION_CREDENTIALS` environment variable with the path to your service account key file:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-file.json"
export GCP_PROJECT_ID="your-gcp-project-id"
```

## How to set up a new crawler using the getter utils

The getter is the process of getting/scraping the data from the web pages, storing the .html files in GCS, and storing metadata
in BigQuery.

When creating a new scraper/collector class, you should inherit from the `AdvisoryCollector` class and implement the `__collect_request_stats()`
method. This method must return a `RequestStatsList` object that contains the metadata to scrap the web pages. 
`RequestStatsList` contains only one attribute (advisories) with a list of `RequestStats` object. When developing the
`__collect_request_stats()` method, at least one attribute of `RequestStats` must be provided: `url`. This will depend on the type of request
that need to be executed:

- For a GET request, the url is usually enough.
- For a POST request, you might need to also add the body, content_type, referer, and authorization to `RequestStats`.

Having a list of `RequestStats` with the correct attribute, allows the getter scrap the URLs, get the .html files, store them in the right
place on GCS, and store metadata in the right BigQuery tables. The `AdvisoryCollector` parent class fills up the rest of the `RequestStats` attributes during the scraping process, like status_code,
timestamp, stored_at, etc. 

Please refer to `security_bulletins_utils.models.py` to check the structure of the `RequestStats` and `RequestStatsList` data classes.

After creating the new scrapper class (e.g., 'AppleScraper'), you should run the scraper by calling the `run_crawler()` method.
This method will scrap the web pages defined in the `RequestStatsList` object and store the data in GCS and BigQuery.

Summing up, create a new class that inherits from `security_bulletins_utils.getter.base.AdvisoryCollector` and implement the
`__collect_request_stats()` method. Then call `run_crawler()` to start the crawling process. 

Here's a dummy example on how to implement a new getter or scraper:

```python
import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, ConfigDict
from security_bulletins_utils.getter.base import AdvisoryCollector
from security_bulletins_utils.models import RequestStatsList, RequestStats
from security_bulletins_utils.storage.gcs import GCSStorage


class MacOsInfo(BaseModel):
    model_config = ConfigDict(extra="allow")
    version: str
    url: str
    html: str


class AppleScraper(AdvisoryCollector):
    def __init__(
        self,
        storage: GCSStorage,
        bigquery_dataset: str,
        gcp_default_credentials: bool = False,
        logger_name: str | None = None,
        proxies: dict[str, str] | None = None,
    ):
        self.base_url: str = 'https://support.apple.com'
        self.home_url: str = 'https://support.apple.com/en-us/100100'

        super().__init__(storage, bigquery_dataset, gcp_default_credentials, logger_name, proxies)

    @property
    def vendor_id(self) -> str:
        return 'Apple'

    def _get_macos_urls(self) -> list[MacOsInfo]:
        response = requests.get(self.home_url)
        soup = BeautifulSoup(response.text, 'html.parser')

        macos_info_list = []

        for link in soup.find_all('a'):
            tags = link.get_text()
            href = link.get('href')

            if 'MACOS' in tags.upper():
                macos_info_list.append(MacOsInfo(version=tags, url=self.base_url + href, html=response.text))
        return macos_info_list

    def _collect_request_stats(self) -> RequestStatsList:
        advisories_to_crawl: list[MacOsInfo] = self._get_macos_urls()
        req_stats_list: RequestStatsList = RequestStatsList()

        for macos_item in advisories_to_crawl:
            req_stats_list.requests.append(
                RequestStats(url=macos_item.url, method='GET', advisory_code=macos_item.version)
            )

        return req_stats_list


# Configure and execute the getter
if __name__ == '__main__':
    # Raw metadata is usually stored in GCS in a folder called 'security-bulletins-advisories-raw'.
    bucket = 'security-bulletins-advisories-raw'

    # Define the GCO storage class with the project id and bucket name. If default_credentials is set to True, it will use the
    # default credentials from the google SDK. If set to False, it will use the service account key file which path is saved in
    # the GOOGLE_APPLICATION_CREDENTIALS environment variable.
    storage = GCSStorage(project_id=os.getenv('GCP_PROJECT_ID'), bucket=bucket, default_credentials=True)

    # Instantiate the scraper class with the storage, bigquery dataset and gcp default credentials. Just like the GCSStorage class,
    # if gcp_default_credentials is set to True, it will use the default credentials from the google SDK. If set to False, it will use
    # the service account key file which path is saved in the GOOGLE_APPLICATION_CREDENTIALS environment variable.
    apple = AppleScraper(storage=storage, bigquery_dataset='<your-dataset-id>', gcp_default_credentials=True)
    
    apple.logger.setLevel(logging.INFO)
    
    # Entrypoint to run the cralwer. The AdvisoryCollector parent class will handle the scraping and storing of the data
    # in GCS and BigQuery.
    apple.run_crawler()
```


## BigQuery Advisories Module

This Python module facilitates interaction with Google BigQuery to manage and process security advisories. It enables querying data,
storing metadata, and marking advisories as processed.

Main functionalities:
- Store advisory metadata and processing history.
- Fetch unparsed advisory paths and mark them as parsed.
- Support for JSON and Pydantic models when handling advisory data.

### Running a query

```python
security_bulletins_utils.bigquery.biguery import BigQueryDatabase

project_id = "ecl-dw-dev"

with BigQueryDatabase(project_id) as db:
    query = "SELECT * FROM dataset.table LIMIT 10"
    result = db.exec_query(query)
```

This method executes a custom SQL query on the specified BigQuery dataset and table.

### Storing metadata in getter table

```python
from security_bulletins_utils.bigquery.bigquery import BigQueryDatabase,
from security_bulletins_utils.models import AdvisoryProcessing

project_id = "ecl-dw-dev"
data = [AdvisoryProcessing(...), AdvisoryProcessing(...), AdvisoryProcessing(...)]

with BigQueryDatabase(project_id) as db:
    db.store_metadata_in_getter_table(data=data)
```

This method stores metadata in the `advisory_getter` table in BigQuery. The default dataset and table values are:
- `dataset`: security_bulletins_utils.bigquery.schemas.ADVISORY_GETTER_DATASET_ID
- `table`: security_bulletins_utils.bigquery.schemas.ADVISORY_GETTER_TABLE_ID

These values can be overridden by passing the `dataset` and `table` arguments.

Each row in the data must be an instance of the `AdvisoryProcessing` model, found in `security_bulletins_utils.bigquery.models`. 
Some fields can be left blank if needed. 

### Fetching unparsed advisories from getter table

```python
from security_bulletins_utils.bigquery.bigquery import BigQueryDatabase

project_id = "ecl-dw-dev"
vendor_name = "Juniper"
target_folder = "Juniper/2024-09-29_00-11-44"

with BigQueryDatabase(project_id) as db:
    advisories = db.get_unparsed_advisories(vendor_name=vendor_name, target_folder=target_folder)
    for advisory in advisories:
        print(advisory.advisory_url, advisory.advisory_located_at)
```

This method retrieves the advisory URLs and their corresponding GCS locations from the `advisory_getter` table,
focusing on entries that have not yet been parsed. The default dataset and table values are:
- `dataset`: security_bulletins_utils.bigquery.schemas.ADVISORY_GETTER_DATASET_ID
- `table`: security_bulletins_utils.bigquery.schemas.ADVISORY_GETTER_TABLE_ID

These values can be overridden by passing the `dataset` and `table` arguments.

The method returns a list of `AdvisoryPath` objects, which are defined in `security_bulletins_utils.bigquery.models`.

### Marking an advisory as parsed in getter table

#### Update one row:

```python
from security_bulletins_utils.bigquery.bigquery import BigQueryDatabase
from security_bulletins_utils.models import AdvisoryPath

project_id = "ecl-dw-dev"

advisory_path = AdvisoryPath(
    advisory_url="https://supportportal.juniper.net/s/article/advisory-url",
    advisory_located_at="Juniper/2024-09-29_00-11-44/filename.html.xz")

with BigQueryDatabase(project_id) as db:
    db.set_advisory_as_parsed(
        vendor_name="Juniper",
        target_folder="Juniper/2024-09-29_00-11-44",
        advisory_path=advisory_path
    )
```

#### Update all rows:

```python
from security_bulletins_utils.bigquery.bigquery import BigQueryDatabase
from security_bulletins_utils.models import AdvisoryPath

project_id = "ecl-dw-dev"

with BigQueryDatabase(project_id) as db:
    db.set_advisory_as_parsed(
        vendor_name="Juniper",
        target_folder="Juniper/2024-09-29_00-11-44"
    )
```

This method updates the `raw_file_parsed` field in the `advisory_processing_history` table to `True` for the rows in the
`advisory_getter` table that match the given `vendor_name` and `target_folder` values.

If the advisory_path argument is not provided, the method will update all unparsed advisories in the `advisory_getter` table for the
given `vendor_name` and `target_folder` values.

If the `advisory_path` argument is provided, the method will update the row in the `advisory_getter` table that matches the
`advisory_url` and `advisory_located_at` values in the `advisory_path` argument.

The default dataset and table values for the `advisory_getter` table are:
- `dataset`: security_bulletins_utils.bigquery.schemas.ADVISORY_GETTER_DATASET_ID
- `table`: security_bulletins_utils.bigquery.schemas.ADVISORY_GETTER_TABLE_ID


## Mongodb Module
### Description
This module enables interaction with MongoDB KDB, leveraging data collected by integrity crawlers for managing security advisories. 
It uses the `custom-mongo-package` library to interface with the MongoDB database KDB.

### Environment Variables
- `MONGO_CERT`: path to the .pem file
- `MONGO_URI`: uri to connect to the mongo database

### Main functionalities
- `get_model_packages`: fetches all the model - packages relations from the mongodb KDB for a given vendor and package type
- `push_advisories_packages`: push the new relations between the advisory and the packages to the mongodb KDB

### Impacted KDB collections
#### INPUT
- `models`: collection that stores all the products (desktop, laptop, server, etc.)
- `updatePackages`: collection that stores the firmware update packages (for UEFI, BIOS, ME, storage, etc.)
- `updatePackages_models`: collection that stores the relations between the models and the packages
#### OUTPUT
- `advisories_updatePackages`: collection that stores the relations between the advisories and the packages

### Example
cf. `security_bulletins_utils/mongodb/example.py`
```