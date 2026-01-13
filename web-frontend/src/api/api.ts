export type ApiConfig = {
  baseUrl: string; // Base URL of my API Gateway
  routePath?: string; // Path for the endpoint; i.e. /image
};

export type ImageUrlResponse = {
  url: string;
};

export type ApiError = {
  status: number;
  message: string;
};

/**
 * Creating an API client for your single API Gateway endpoint.
 */
export function createApiClient(config: ApiConfig) {
  const routePath = config.routePath ?? "/image";

  function buildUrl(param: string) {
    const u = new URL(config.baseUrl.replace(/\/$/, "") + routePath + param);
    return u.toString();
  }

  async function fetchJson<T>(url: string): Promise<T> {
    const res = await fetch(url, { method: "GET" });

    if (!res.ok) {
      const text = await res.text().catch(() => "");
      const msg = text || `Request failed with status ${res.status}`;
      throw { status: res.status, message: msg } satisfies ApiError;
    }

    return (await res.json()) as T;
  }

  /**
   * Generic call: gets the public S3 URL for any key.
   * This expects your Lambda to return JSON: { "url": "https://..." }.
   */
  async function getPublicUrlForKey(parameter: string): Promise<string> {
    const url = buildUrl(parameter);
    console.log("GET: ", url)
    const data = await fetchJson<ImageUrlResponse>(url);

    if (!data?.url) {
      throw { status: 500, message: "API response missing `url`" } satisfies ApiError;
    }

    return data.url;
  }

  function getInputImageUrl(imageName: string) {
      return getPublicUrlForKey(`?key=input-images/${imageName}`);
  }

  function getOutputImageUrl(imageName: string) {
      return getPublicUrlForKey(`?key=output-images/${imageName}`);
  }

  function getOutputPredsUrl(jsonName: string) {
      return getPublicUrlForKey(`?key=output-preds/${jsonName}`);
  }



  return {
    getInputImageUrl,
    getOutputImageUrl,
    getOutputPredsUrl,
  };
}
