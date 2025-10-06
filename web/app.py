"""SmartDiet Gradio Web Interface"""

import os
import json
import requests
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import pandas as pd
import gradio as gr


PARENT_DIR = Path(__file__).parent.resolve()
DATA_DIR = PARENT_DIR / "data"

# Configuration
API_URL = os.getenv("SMARTDIET_API_URL", "http://localhost:8000")
PRODUCTS_CSV = DATA_DIR / "products_filtered_with_images_en.csv"
IMAGE_DIR = DATA_DIR / "images"
RESTRICTIONS_FILE = PARENT_DIR / "web" / "user_restrictions.json"


class ProductDatabase:
    """Handle product data loading and queries."""

    def __init__(self, csv_path: Path, image_dir: Path):
        """Load product database from CSV."""
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        print(f"Loaded {len(self.df)} products from database")

    def find_product_image(self, product_id: str) -> Optional[str]:
        """Find product image by ID in the image directory."""
        # Try common image extensions
        extensions = ['.png', '.jpg', '.jpeg', '.webp']
        for ext in extensions:
            image_path = self.image_dir / f"{product_id}{ext}"
            if image_path.exists():
                return str(image_path)
        return None

    def get_product_info(self, product_id: str) -> Optional[Dict]:
        """Get product information by ID."""
        product = self.df[self.df['id'] == product_id]
        if product.empty:
            return None

        row = product.iloc[0]
        # Find actual image file
        image_path = self.find_product_image(product_id)

        return {
            'id': row['id'],
            'title': row['title_en'],
            'description': row['description_en'],
            'ingredients': row['list_of_ingredients_en'],
            'calories': row['calories'],
            'carbohydrates': row['carbohydrates'],
            'proteins': row['proteins'],
            'fats': row['fats'],
            'category': row['category'],
            'image': row['image'],
            'image_path': image_path,  # Local file path
            'product_link': row['product_link']
        }


class RestrictionManager:
    """Manage user food restrictions with persistence."""

    def __init__(self, storage_path: Path):
        """Initialize restriction manager."""
        self.storage_path = storage_path
        self.storage_path.parent.mkdir(exist_ok=True, parents=True)

    def load_restrictions(self) -> str:
        """Load saved restrictions from file."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get('restrictions', '')
            except Exception as e:
                print(f"Error loading restrictions: {e}")
        return ""

    def save_restrictions(self, restrictions: str) -> None:
        """Save restrictions to file."""
        try:
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump({'restrictions': restrictions}, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving restrictions: {e}")

    def parse_restrictions(self, restrictions_text: str) -> List[str]:
        """Parse comma-separated restrictions into list."""
        if not restrictions_text or not restrictions_text.strip():
            return []
        return [r.strip().lower() for r in restrictions_text.split(',') if r.strip()]


class SmartDietAPI:
    """Interface to SmartDiet ML API."""

    def __init__(self, base_url: str):
        """Initialize API client."""
        self.base_url = base_url

    def search_similar_products(self, image_path: str, top_k: int = 5) -> List[Dict]:
        """Search for similar products by image."""
        try:
            # Determine file extension and content type
            from pathlib import Path
            ext = Path(image_path).suffix.lower()

            content_type_map = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.webp': 'image/webp'
            }
            content_type = content_type_map.get(ext, 'image/jpeg')

            with open(image_path, 'rb') as f:
                # Provide proper filename and content type
                files = {
                    'file': (f'image{ext}', f, content_type)
                }
                params = {'top_k': top_k}
                response = requests.post(
                    f"{self.base_url}/search",
                    files=files,
                    params=params,
                    timeout=30
                )

            if response.status_code == 200:
                data = response.json()
                return data.get('results', [])
            else:
                print(f"API error: {response.status_code} - {response.text}")
                return []
        except Exception as e:
            print(f"Error calling API: {e}")
            return []


class SmartDietApp:
    """Main Gradio application."""

    def __init__(self):
        """Initialize the application."""
        self.api = SmartDietAPI(API_URL)
        self.products = ProductDatabase(PRODUCTS_CSV, IMAGE_DIR)
        self.restrictions = RestrictionManager(RESTRICTIONS_FILE)
        self.current_search_results = []

    def search_products(self, image, restrictions_text: str) -> Tuple[str, str, str]:
        """Search for products and return top 5 for selection."""
        if image is None:
            return "Please upload an image first.", "", restrictions_text

        # Save restrictions
        self.restrictions.save_restrictions(restrictions_text)

        # Search for similar products
        results = self.api.search_similar_products(image, top_k=5)

        if not results:
            return "No products found. Make sure the API is running.", "", restrictions_text

        # Store results for later selection
        self.current_search_results = results

        # Format results for display
        output = "## Top 5 Similar Products\n\n"
        output += "Select a product number (1-5) below to see detailed information and restriction check.\n\n"

        for i, result in enumerate(results, 1):
            product_info = self.products.get_product_info(result['product_id'])
            if product_info:
                output += f"**{i}. {product_info['title']}**\n"
                output += f"   - Similarity: {result['similarity']:.2%}\n"
                output += f"   - Category: {product_info['category']}\n\n"

        return output, "Search completed! Select a product number (1-5) to see details.", restrictions_text

    def check_restrictions(self, product_info: Dict, restrictions_list: List[str]) -> Tuple[bool, List[str]]:
        """Check if product contains any restricted ingredients."""
        if not restrictions_list:
            return True, []

        ingredients = str(product_info.get('ingredients', '')).lower()
        title = str(product_info.get('title', '')).lower()
        description = str(product_info.get('description', '')).lower()

        # Check for restrictions in ingredients, title, and description
        found_restrictions = []
        for restriction in restrictions_list:
            if (restriction in ingredients or
                restriction in title or
                restriction in description):
                found_restrictions.append(restriction)

        is_safe = len(found_restrictions) == 0
        return is_safe, found_restrictions

    def select_product(self, product_number: str, restrictions_text: str) -> Tuple[str, str, Optional[str]]:
        """Display detailed product info and restriction check."""
        try:
            # Parse product number
            idx = int(product_number) - 1
            if idx < 0 or idx >= len(self.current_search_results):
                return "Invalid product number. Please enter a number between 1 and 5.", "", None

            # Get product info
            result = self.current_search_results[idx]
            product_info = self.products.get_product_info(result['product_id'])

            if not product_info:
                return "Product information not found.", "", None

            # Parse restrictions
            restrictions_list = self.restrictions.parse_restrictions(restrictions_text)

            # Check restrictions
            is_safe, found_restrictions = self.check_restrictions(product_info, restrictions_list)

            # Format output
            output = f"## {product_info['title']}\n\n"
            output += f"**Description:** {product_info['description']}\n\n"
            output += f"**Ingredients:** {product_info['ingredients']}\n\n"

            # Nutritional information
            output += "### Nutritional Information (per 100g)\n"
            if product_info['calories'] > 0:
                output += f"- **Calories:** {product_info['calories']} kcal\n"
            if product_info['proteins'] > 0:
                output += f"- **Proteins:** {product_info['proteins']}g\n"
            if product_info['carbohydrates'] > 0:
                output += f"- **Carbohydrates:** {product_info['carbohydrates']}g\n"
            if product_info['fats'] > 0:
                output += f"- **Fats:** {product_info['fats']}g\n"

            output += f"\n**Category:** {product_info['category']}\n"
            output += f"**Similarity Score:** {result['similarity']:.2%}\n\n"

            # Restriction result
            if is_safe:
                verdict = "# <span style=\"color:green;\">✅ SAFE TO CONSUME</span>"
                verdict += "This product does not contain any of your restricted ingredients."
            else:
                verdict = "# <span style=\"color:red;\">❌ NOT RECOMMENDED</span>\n\n"
                verdict += f"This product contains the following restricted ingredients:\n"
                for restriction in found_restrictions:
                    verdict += f"- {restriction}\n"

            # Get product image path
            image_path = product_info.get('image_path')

            return output, verdict, image_path

        except ValueError:
            return "Please enter a valid number (1-5).", "", None
        except Exception as e:
            return f"Error processing product: {str(e)}", "", None

    def create_interface(self) -> gr.Blocks:
        """Create Gradio interface."""

        # Load saved restrictions
        saved_restrictions = self.restrictions.load_restrictions()

        with gr.Blocks(title="SmartDiet - Food Restriction Checker", theme=gr.themes.Base()) as app:
            # Custom JavaScript for localStorage persistence
            gr.HTML("""
                <script>
                    // Load dietary restrictions from localStorage on page load
                    function loadRestrictions() {
                        const saved = localStorage.getItem('smartdiet_restrictions');
                        if (saved) {
                            const textbox = document.querySelector('textarea[placeholder*="milk, eggs"]');
                            if (textbox) {
                                textbox.value = saved;
                                textbox.dispatchEvent(new Event('input', { bubbles: true }));
                            }
                        }
                    }

                    // Save dietary restrictions to localStorage on input
                    function setupRestrictionsSaving() {
                        const textbox = document.querySelector('textarea[placeholder*="milk, eggs"]');
                        if (textbox) {
                            textbox.addEventListener('input', function() {
                                localStorage.setItem('smartdiet_restrictions', this.value);
                            });
                            textbox.addEventListener('change', function() {
                                localStorage.setItem('smartdiet_restrictions', this.value);
                            });
                        }
                    }

                    // Initialize after DOM is ready
                    if (document.readyState === 'loading') {
                        document.addEventListener('DOMContentLoaded', function() {
                            setTimeout(function() {
                                loadRestrictions();
                                setupRestrictionsSaving();
                            }, 500);
                        });
                    } else {
                        setTimeout(function() {
                            loadRestrictions();
                            setupRestrictionsSaving();
                        }, 500);
                    }
                </script>
            """)

            gr.Markdown("""# 🥗 SmartDiet - Food Restriction Checker

                Upload a food product image to find similar products and check if they match your dietary restrictions.

                ### How to use:

                1. Enter your dietary restrictions (comma-separated, e.g., "milk, nuts, gluten")
                2. Upload an image of a food product
                3. Click "Search Products" to find top 5 similar items
                4. Select a product number (1–5) to see details and restriction check

                ### 📋 Important:

                Before using the application, please **complete the mandatory survey** at the following link:
                👉 [https://forms.gle/wQ1RMYbibNfkjsjv6](https://forms.gle/wQ1RMYbibNfkjsjv6)

                Your feedback helps us improve SmartDiet and tailor it to users' needs.
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    restrictions_input = gr.Textbox(
                        label="🚫 Dietary Restrictions (comma-separated)",
                        placeholder="e.g., milk, eggs, nuts, gluten, sugar",
                        value=saved_restrictions,
                        lines=3,
                        info="Your restrictions will be saved automatically"
                    )

                    image_input = gr.Image(
                        label="📸 Upload Food Product Image",
                        type="filepath",
                        height=300
                    )

                    search_btn = gr.Button("🔍 Search Products", variant="primary", size="lg")

                    gr.Markdown("---")

                    product_number = gr.Textbox(
                        label="Select Product Number (1-5)",
                        placeholder="Enter 1, 2, 3, 4, or 5",
                        max_lines=1
                    )

                    select_btn = gr.Button("✅ Check This Product", variant="secondary", size="lg")

                with gr.Column(scale=2):
                    status_output = gr.Markdown(label="Status")

                    search_results = gr.Markdown(
                        label="Search Results",
                        value="Upload an image and click 'Search Products' to begin."
                    )

                    gr.Markdown("---")

                    restriction_verdict = gr.Markdown(label="Restriction Check")

                    product_image = gr.Image(
                        label="📦 Product Image",
                        type="filepath",
                        height=350,
                        show_label=True
                    )

                    product_details = gr.Markdown(label="Product Details")


            # Event handlers
            search_btn.click(
                fn=self.search_products,
                inputs=[image_input, restrictions_input],
                outputs=[search_results, status_output, restrictions_input]
            )

            select_btn.click(
                fn=self.select_product,
                inputs=[product_number, restrictions_input],
                outputs=[product_details, restriction_verdict, product_image]
            )

        return app


def main():
    """Main entry point."""
    app = SmartDietApp()
    interface = app.create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )


if __name__ == "__main__":
    main()
