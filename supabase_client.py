"""Supabase client configuration loaded from protected environment variables."""

import os 
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()  # Load environment variables from .env file

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

secret_key = os.getenv("SECRET_KEY")