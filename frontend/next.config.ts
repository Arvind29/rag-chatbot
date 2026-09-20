import type { NextConfig } from "next";
const nextConfig: NextConfig = { async rewrites(){ if(process.env.NODE_ENV!=="development")return []; const target=process.env.LOCAL_API_URL||"http://127.0.0.1:8005"; return [{source:"/api/:path*",destination:`${target}/api/:path*`}]; } };
export default nextConfig;
