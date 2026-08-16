"""Authentication and user-profile routes backed by Supabase Auth."""

from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from supabase import create_client, Client
from supabase_client import supabase  # Import the Supabase client from supabase_client.py
from datetime import datetime

supabase: Client = supabase  # Type hint for the Supabase client
auth = Blueprint('auth', __name__)

@auth.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        email = request.form.get("email")
        password = request.form.get("password")
        username = request.form.get("username", "")
        # username = request.form.get("username")

        try:
            result = supabase.auth.sign_up({
                "email": email,
                "password": password
            })

            user = result.user # generated user id from supabase

            if user:
                supabase.table("profiles").insert({
                    "id": user.id,
                    "email": email,
                    "username" : username if username else email.split('@')[0]  # Use email prefix as username if not provided
                    # "username": username
                }).execute()

                flash("Registration successful! Please login.", "success")
                return redirect(url_for('auth.login'))

        except Exception as e:
            error_msg = str(e)
            if "already registered" in error_msg.lower():
                flash("Email already registered. Please login.", "danger")
            elif "duplicate key" in error_msg.lower():
                flash("Username already taken. Please choose another.", "warning")
            else:
                flash(f"Registration error: {error_msg}", "danger")
            print(f"Registration error: {error_msg}")

    return render_template("register.html")

@auth.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        email = request.form["email"]
        password = request.form["password"]
        # username = request.form.get("username")

        try:
            result = supabase.auth.sign_in_with_password({
                "email": email,
                "password": password
            })

            # supabase.table("profiles").update({
            #     "last_login": datetime.now().isoformat()
            # }).eq("id", result.user.id).execute()
            
            session["access_token"] = result.session.access_token
            session["refresh_token"] = result.session.refresh_token
            session["user_id"] = result.user.id
            session["email"] = result.user.email
            session["logged_in"] = True  

            supabase.auth.set_session(
                result.session.access_token,
                result.session.refresh_token
            )
            flash("Login successful!", "success")
            return redirect(url_for('index'))

        except Exception as e:
            error_msg = str(e)
            print(f"Login error: {error_msg}")  # Debug
            
            if "rate limit" in error_msg.lower() or "too many requests" in error_msg.lower():
                flash("Too many login attempts. Please wait a few minutes.", "danger")
            elif "invalid credentials" in error_msg.lower() or "invalid login" in error_msg.lower():
                flash("Invalid email or password. Please try again.", "danger")
            else:
                flash(f"Login error: {error_msg}", "danger")

    return render_template("login.html")

@auth.route("/logout")
def logout():
    try:
        supabase.auth.sign_out()
    except:
        pass
    
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for('auth.login'))

@auth.route("/profile", methods=["GET", "POST"])
def profile():
    if "access_token" not in session:
        flash("Please login first.", "warning")
        return redirect(url_for('auth.login'))
    
    if request.method == "POST":
        try:
            username = request.form.get("username")
            full_name = request.form.get("full_name")
            
            update_data = {}
            if username:
                update_data["username"] = username
            if full_name:
                update_data["full_name"] = full_name
            update_data["updated_at"] = datetime.now().isoformat()
            
            if update_data:
                supabase.table("profiles")\
                    .update(update_data)\
                    .eq("id", session.get("user_id"))\
                    .execute()
                
                flash("Profile updated successfully!", "success")
            
        except Exception as e:
            flash(f"Error updating profile: {str(e)}", "danger")
        
        return redirect(url_for('auth.profile'))
    
    # Get current profile
    try:
        profile = supabase.table("profiles")\
            .select("*")\
            .eq("id", session.get("user_id"))\
            .execute()
        
        user_data = profile.data[0] if profile.data else {}
    except:
        user_data = {}
    
    return render_template("profile.html", profile=user_data, logged_in=True)