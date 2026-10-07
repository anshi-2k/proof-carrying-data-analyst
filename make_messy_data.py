"""Creates a small multi-table dataset full of traps in data/."""
import os

os.makedirs("data", exist_ok=True)

orders = '''Order_ID,Order_Date,Product,Region,Quantity,Sales,Profit
1001,03/04/2026,Laptop,North,2,"$2,400",300
1002,05/04/2026,Mouse,north,10,$150,40
1003,04/05/2026,Phone,South,3,€2100,250
1004,06/05/2026,Laptop,East,1,$1200,150
1004,06/05/2026,Laptop,East,1,$1200,150
1005,07/05/2026,Desk,West,2,N/A,
1006,08/06/2026,Chair,East,4,$600,90
1007,09/06/2026,Mouse,South,5,$75,20
1007,09/06/2026,Mouse,South,5,$750,20
'''
products = '''Product,Category,List_Price
Laptop,Electronics,1200
Mouse,Electronics,15
Phone,Electronics,700
Desk,Furniture,300
Chair,Furniture,150
'''
# North=12 correct, South should be 13 (contradiction), East=5, West=2
region_summary = '''Region,Total_Quantity
North,12
South,8
East,5
West,2
'''
for name, text in (("orders", orders), ("products", products), ("region_summary", region_summary)):
    with open(f"data/{name}.csv", "w", encoding="utf-8") as f:
        f.write(text)
print("Wrote data/orders.csv, data/products.csv, data/region_summary.csv")
