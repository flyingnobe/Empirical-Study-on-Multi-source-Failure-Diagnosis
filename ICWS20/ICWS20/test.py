string = "('productcatalogservice', 'currencyservice')"
start_index = string.find("'") + 1
end_index = string.find("'", start_index)
content = string[start_index:end_index]
print(content)
