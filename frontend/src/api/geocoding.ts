export interface LocationSuggestion {
  id: string;
  name: string;
  secondaryText: string;
  displayName: string;
  center: [number, number];
  zoom: number;
}

function calculateZoomFromSpan(maxSpan: number): number {
  if (maxSpan > 25) return 4;
  if (maxSpan > 12) return 5;
  if (maxSpan > 5) return 7;
  if (maxSpan > 2) return 8;
  if (maxSpan > 0.8) return 10;
  if (maxSpan > 0.2) return 11;
  if (maxSpan > 0.05) return 12;
  return 13;
}

function calculateZoom(extent?: number[], type?: string): number {
  if (extent && extent.length === 4) {
    const lngSpan = Math.abs(extent[2] - extent[0]);
    const latSpan = Math.abs(extent[3] - extent[1]);
    return calculateZoomFromSpan(Math.max(lngSpan, latSpan));
  }

  if (type === "country") return 5;
  if (type === "state" || type === "region") return 7;
  if (type === "county" || type === "district") return 9;
  if (type === "city" || type === "town") return 11;
  return 11;
}

/**
 * Searches for geographic locations with typeahead autocomplete suggestions.
 * Uses Photon (OSM) as primary with OpenStreetMap Nominatim fallback.
 */
export async function searchLocations(
  query: string,
  signal?: AbortSignal
): Promise<LocationSuggestion[]> {
  const trimmed = query.trim();
  if (trimmed.length < 2) {
    return [];
  }

  // 1. Try Photon (specifically optimized for search-as-you-type autocomplete)
  try {
    const photonUrl = `https://photon.komoot.io/api/?q=${encodeURIComponent(trimmed)}&limit=5`;
    const response = await fetch(photonUrl, {
      method: "GET",
      headers: {
        Accept: "application/json",
      },
      signal,
    });

    if (response.ok) {
      const data = await response.json();
      const features = Array.isArray(data?.features) ? data.features : [];

      if (features.length > 0) {
        return features.slice(0, 5).map((feature: any, index: number) => {
          const props = feature.properties || {};
          const name = props.name || trimmed;

          const contextParts: string[] = [];
          if (props.city && props.city.toLowerCase() !== name.toLowerCase()) {
            contextParts.push(props.city);
          }
          if (
            props.county &&
            props.county.toLowerCase() !== name.toLowerCase() &&
            !contextParts.includes(props.county)
          ) {
            contextParts.push(props.county);
          }
          if (
            props.state &&
            props.state.toLowerCase() !== name.toLowerCase() &&
            !contextParts.includes(props.state)
          ) {
            contextParts.push(props.state);
          }
          if (props.country && props.country.toLowerCase() !== name.toLowerCase()) {
            contextParts.push(props.country);
          }

          const secondaryText = contextParts.join(", ");
          const displayName = secondaryText ? `${name}, ${secondaryText}` : name;
          const coords = feature.geometry?.coordinates || [0, 0];
          const center: [number, number] = [coords[1], coords[0]];
          const zoom = calculateZoom(props.extent, props.type);

          return {
            id: `photon-${props.osm_id || index}-${coords[0]}-${coords[1]}`,
            name,
            secondaryText,
            displayName,
            center,
            zoom,
          };
        });
      }
    }
  } catch (err: any) {
    if (err?.name === "AbortError") {
      throw err;
    }
    // Fall back to Nominatim below
  }

  // 2. Fallback to OpenStreetMap Nominatim
  try {
    const nominatimUrl = `https://nominatim.openstreetmap.org/search?q=${encodeURIComponent(
      trimmed
    )}&format=json&addressdetails=1&limit=5`;

    const response = await fetch(nominatimUrl, {
      method: "GET",
      headers: {
        Accept: "application/json",
      },
      signal,
    });

    if (response.ok) {
      const data = await response.json();
      if (Array.isArray(data) && data.length > 0) {
        return data.slice(0, 5).map((item: any, index: number) => {
          const name = item.name || (item.display_name ? item.display_name.split(",")[0] : trimmed);
          const rawParts = item.display_name
            ? item.display_name.split(",").map((s: string) => s.trim())
            : [];
          const secondaryParts = rawParts
            .slice(1)
            .filter((part: string) => part && !/^\d{4,6}$/.test(part));
          const secondaryText = secondaryParts.join(", ");
          const displayName = secondaryText ? `${name}, ${secondaryText}` : name;

          const lat = parseFloat(item.lat);
          const lon = parseFloat(item.lon);
          const center: [number, number] = [lat, lon];

          let zoom = 11;
          if (Array.isArray(item.boundingbox) && item.boundingbox.length === 4) {
            const south = parseFloat(item.boundingbox[0]);
            const north = parseFloat(item.boundingbox[1]);
            const west = parseFloat(item.boundingbox[2]);
            const east = parseFloat(item.boundingbox[3]);
            zoom = calculateZoomFromSpan(Math.max(Math.abs(north - south), Math.abs(east - west)));
          } else if (item.type === "administrative" || item.class === "boundary") {
            zoom = 7;
          }

          return {
            id: `nom-${item.place_id || index}-${lat}-${lon}`,
            name,
            secondaryText,
            displayName,
            center,
            zoom,
          };
        });
      }
    }
  } catch (err: any) {
    if (err?.name === "AbortError") {
      throw err;
    }
    console.warn("Geocoding lookup failed:", err);
  }

  return [];
}
