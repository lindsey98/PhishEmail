from SPARQLWrapper import SPARQLWrapper, JSON

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
