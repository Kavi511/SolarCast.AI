import { Mail, Linkedin, Github, Instagram, Youtube } from "lucide-react";
import { useLocation } from "react-router-dom";

const Footer = () => {
  const location = useLocation();
  const isAuthPage = location.pathname === '/login' || location.pathname === '/register';
  return (
    <footer className="bg-gray-900 border-t border-gray-800 py-12 dark">
      <div className="container mx-auto px-6">
        <div className={
          isAuthPage
            ? "grid grid-cols-1 gap-8 place-items-center text-center"
            : "grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8"
        }>
          
          {/* Platform Identity */}
          <div className="space-y-4">
            <h3 className="text-xl font-bold text-slate-100">SolarCast.AI</h3>
            <p className="text-slate-300">
              Predicting solar potential with GIS & AI.
            </p>
            <div className="pt-4">
              <p className="text-sm text-slate-400">
                © 2025 SolarCast.AI | All Rights Reserved
              </p>
            </div>
          </div>

          {/* Quick Links */}
          <div className="space-y-4">
            <h4 className="text-lg font-semibold text-slate-100">Quick Links</h4>
            <ul className="space-y-2">
              <li>
                <a href="#about" className="text-slate-300 hover:text-slate-100 transition-colors">
                  About
                </a>
              </li>
              <li>
                <a href="#contact" className="text-slate-300 hover:text-slate-100 transition-colors">
                  Contact
                </a>
              </li>
              <li>
                <a href="#map-viewer" className="text-slate-300 hover:text-slate-100 transition-colors">
                  Map Viewer
                </a>
              </li>
              <li>
                <a href="#data-catalog" className="text-slate-300 hover:text-slate-100 transition-colors">
                  Data Catalog
                </a>
              </li>
              <li>
                <a href="#research" className="text-slate-300 hover:text-slate-100 transition-colors">
                  Research / Publications
                </a>
              </li>
            </ul>
          </div>

          {/* Legal & Policies */}
          <div className="space-y-4">
            <h4 className="text-lg font-semibold text-slate-100">Legal & Policies</h4>
            <ul className="space-y-2">
              <li>
                <a href="#privacy" className="text-slate-300 hover:text-slate-100 transition-colors">
                  Privacy Policy
                </a>
              </li>
              <li>
                <a href="#terms" className="text-slate-300 hover:text-slate-100 transition-colors">
                  Terms of Use
                </a>
              </li>
              <li>
                <a href="#licensing" className="text-slate-300 hover:text-slate-100 transition-colors">
                  Data Licensing / Disclaimer
                </a>
              </li>
            </ul>
          </div>

          {/* Contact & Support */}
          <div className="space-y-4">
            <h4 className="text-lg font-semibold text-slate-100">Contact & Support</h4>
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Mail className="w-4 h-4 text-blue-400" />
                <a href="mailto:support@solarcast.ai" className="text-slate-300 hover:text-slate-100 transition-colors text-sm">
                  support@solarcast.ai
                </a>
              </div>
              <div className="flex items-center gap-2">
                <Mail className="w-4 h-4 text-blue-400" />
                <a href="mailto:contact@solarcast.ai" className="text-slate-300 hover:text-slate-100 transition-colors text-sm">
                  contact@solarcast.ai
                </a>
              </div>
            </div>
            
            {/* Social Media */}
            <div className={isAuthPage ? "flex items-center gap-4 pt-2 justify-center" : "flex items-center gap-4 pt-2"}>
              <a href="https://x.com" className="text-slate-300 hover:text-slate-100 transition-colors">
                <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"/>
                </svg>
              </a>
              <a href="#" className="text-slate-300 hover:text-slate-100 transition-colors">
                <Instagram className="w-5 h-5" />
              </a>
              <a href="#" className="text-slate-300 hover:text-slate-100 transition-colors">
                <Github className="w-5 h-5" />
              </a>
              <a href="#" className="text-slate-300 hover:text-slate-100 transition-colors">
                <Linkedin className="w-5 h-5" />
              </a>
              <a href="#" className="text-slate-300 hover:text-slate-100 transition-colors">
                <Youtube className="w-5 h-5" />
              </a>
              <a href="#" className="text-slate-300 hover:text-slate-100 transition-colors">
                <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M20.317 4.37a19.791 19.791 0 0 0-4.885-1.515a.074.074 0 0 0-.079.037c-.21.375-.444.864-.608 1.25a18.27 18.27 0 0 0-5.487 0a12.64 12.64 0 0 0-.617-1.25a.077.077 0 0 0-.079-.037A19.736 19.736 0 0 0 3.677 4.37a.07.07 0 0 0-.032.027C.533 9.046-.32 13.58.099 18.057a.082.082 0 0 0 .031.057a19.9 19.9 0 0 0 5.993 3.03a.078.078 0 0 0 .084-.028a14.09 14.09 0 0 0 1.226-1.994a.076.076 0 0 0-.041-.106a13.107 13.107 0 0 1-1.872-.892a.077.077 0 0 1-.008-.128a10.2 10.2 0 0 0 .372-.292a.074.074 0 0 1 .077-.010c3.928 1.793 8.18 1.793 12.062 0a.074.074 0 0 1 .078.01c.12.098.246.198.373.292a.077.077 0 0 1-.006.127a12.299 12.299 0 0 1-1.873.892a.077.077 0 0 0-.041.107c.36.698.772 1.362 1.225 1.993a.076.076 0 0 0 .084.028a19.839 19.839 0 0 0 6.002-3.03a.077.077 0 0 0 .032-.054c.5-5.177-.838-9.674-3.549-13.66a.061.061 0 0 0-.031-.03zM8.02 15.33c-1.183 0-2.157-1.085-2.157-2.419c0-1.333.956-2.419 2.157-2.419c1.21 0 2.176 1.096 2.157 2.42c0 1.333-.956 2.418-2.157 2.418zm7.975 0c-1.183 0-2.157-1.085-2.157-2.419c0-1.333.955-2.419 2.157-2.419c1.210 0 2.176 1.096 2.157 2.42c0 1.333-.946 2.418-2.157 2.418z"/>
                </svg>
              </a>
            </div>
          </div>
        </div>

        {/* GIS-Specific Info */}
        <div className="border-t border-slate-700 mt-8 pt-8">
          <div className={isAuthPage ? "grid grid-cols-1 gap-6 text-center" : "grid grid-cols-1 md:grid-cols-2 gap-6"}>
            <div>
              <h5 className="text-sm font-semibold text-slate-100 mb-2">Data Sources</h5>
              <div className="space-y-1 text-xs text-slate-400">
                <p>Map data © OpenStreetMap contributors</p>
                <p>Weather data © OpenWeatherMap</p>
                <p>Satellite imagery © Google Earth Engine</p>
              </div>
            </div>
            <div>
              <h5 className="text-sm font-semibold text-slate-100 mb-2">Technical Info</h5>
              <div className="space-y-1 text-xs text-slate-400">
                <p>Projection: WGS 84 (EPSG:4326)</p>
                <p>Last updated: August 2025</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
