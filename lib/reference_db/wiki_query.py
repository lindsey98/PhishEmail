import requests
import pandas as pd
from typing import Dict, Union

def return_sparql_query_results(
    query_string: str, wikidata_sparql_url: str = "https://query.wikidata.org/sparql", proxies:Union[float, str]=None
) -> Dict:
    """Send a SPARQL query and return the JSON formatted result.

    Parameters
    ----------
    query_string: str
      SPARQL query string
    wikidata_sparql_url: str, optional
      wikidata SPARQL endpoint to use
    """
    if proxies:
        return requests.get(
            wikidata_sparql_url,
            proxies=proxies,
            params={"query": query_string, "format": "json"}
        ).json()
    else:
        return requests.get(
            wikidata_sparql_url, params={"query": query_string, "format": "json"}
        ).json()

def get_contact_emails(entity_id):
    """Retrieves contact emails associated with a Wikidata entity.

    Args:
        entity_id (str): The Wikidata entity ID (e.g., "Q12345")

    Returns:
        list: A list of contact emails found.
    """

    query_string = f"""
    SELECT ?email 
    WHERE {{
      wd:{entity_id} schema:contactPoint ?contactPoint . 
      ?contactPoint schema:email ?email . 
    }} 
    """

    r = return_sparql_query_results(query_string, proxies={
        "http": "http://127.0.0.1:7890",
        "https": "http://127.0.0.1:7890",
    })

    emails = []
    for binding in r["results"]["bindings"]:
        emails.append(binding["email"]["value"])

    return emails


entity_id = "Q355"  # Example: Brand as an instance_of
emails = get_contact_emails(entity_id)

if emails:
    print("Contact Emails Found:", emails)
else:
    print("No contact emails found for this entity.")