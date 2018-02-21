"""
Script for using Wikipedia as distant supervision,alternative to Probase. Probase is more useful.
"""
import wikipedia

query = "bebe"

pages = wikipedia.search(query)
for pageName in pages:
    page = wikipedia.page(pageName)
    content = page.content.lower()
    if "brand" in content:
        print "match brand"
        print page.title

bebe = wikipedia.page("Bebe Stores")