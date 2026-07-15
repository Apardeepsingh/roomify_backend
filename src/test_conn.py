from config import supabase

res = supabase.table("products").select("id").execute()
print("connected, rows:", len(res.data)) 