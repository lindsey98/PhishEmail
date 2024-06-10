from SPARQLWrapper import SPARQLWrapper, JSON
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


def fetch_brand_data(brand_name):
    # Ensure the brand_name is safely formatted for inclusion in a SPARQL query
    safe_brand_name = brand_name.replace('"', '\\"').replace("'", "\\'")

    # Define the SPARQL query with dynamic brand input
    query = f"""
    SELECT ?brand ?brandLabel ?description ?officialWebsite 
    WHERE {{
      # Search for entities with the specified brand name
      SERVICE wikibase:mwapi {{
        bd:serviceParam wikibase:endpoint "www.wikidata.org";
                        wikibase:api "EntitySearch";
                        mwapi:search "{safe_brand_name}";
                        mwapi:language "en".
        ?brand wikibase:apiOutputItem mwapi:item.
      }}

      # Optional: fetch description in English
      OPTIONAL {{ ?brand schema:description ?description FILTER(LANG(?description) = "en") }}

      # Optional: fetch official website
      OPTIONAL {{ ?brand wdt:P856 ?officialWebsite }}

      # Label service to get labels of brands and countries in English
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    LIMIT 1
    """

    # Set up the SPARQL connection
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    # Execute the query and print the results
    wikiID = ""
    description = ""
    officialWebsite = ""
    OfficialName = ""
    try:
        results = sparql.query().convert()
        for result in results["results"]["bindings"]:
            OfficialName = result["brandLabel"]["value"]
            wikiID = result["brand"]["value"]
            if "description" in result:
                description = result["description"]["value"]
            if "officialWebsite" in result:
                officialWebsite = result["officialWebsite"]["value"]
    except Exception as e:
        print("An error occurred:", e)

    print(brand_name, f"Wikidata name = {OfficialName}")
    return OfficialName, wikiID, description, officialWebsite


if __name__ == '__main__':

    entity_id = "Q355"  # Example: Brand as an instance_of
    emails = get_contact_emails(entity_id)

    if emails:
        print("Contact Emails Found:", emails)
    else:
        print("No contact emails found for this entity.")