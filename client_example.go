// Клиент к мультиязычному сервису Pora AI для Go-бэкенда.
// Скопируйте в пакет (напр. internal/ai). Прокидывайте локаль пользователя в lang.
package ai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

type Client struct {
	baseURL string
	http    *http.Client
}

func New(baseURL string) *Client {
	return &Client{baseURL: baseURL, http: &http.Client{Timeout: 35 * time.Second}}
}

func (c *Client) post(ctx context.Context, path string, in, out any) error {
	body, _ := json.Marshal(in)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+path, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.http.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		return fmt.Errorf("pora-ai %s: status %d", path, resp.StatusCode)
	}
	return json.NewDecoder(resp.Body).Decode(out)
}

// ---------- /v1/replenishment ----------
type Purchase struct {
	Product string `json:"product"`
	Date    string `json:"date"` // ISO 2006-01-02
}
type Prediction struct {
	Product    string  `json:"product"`
	EveryDays  float64 `json:"every_days"`
	DueDate    string  `json:"due_date"`
	DaysLeft   int     `json:"days_left"`
	Confidence float64 `json:"confidence"`
	Status     string  `json:"status"`
}

func (c *Client) Replenishment(ctx context.Context, today string, p []Purchase) ([]Prediction, error) {
	var out struct {
		Predictions []Prediction `json:"predictions"`
	}
	err := c.post(ctx, "/v1/replenishment", map[string]any{"today": today, "purchases": p}, &out)
	return out.Predictions, err
}

// ---------- /v1/categorize (мультиязычно) ----------
type CategoryResult struct {
	Name         string  `json:"name"`
	Section      string  `json:"section"`       // ключ: dairy/produce/...
	SectionLabel string  `json:"section_label"` // локализованное название
	Confidence   float64 `json:"confidence"`
	Lang         string  `json:"lang"`
	Method       string  `json:"method"` // fast | llm
}

// lang — локаль пользователя ("ru","en",...); "" = автоопределение
func (c *Client) Categorize(ctx context.Context, names []string, lang string) ([]CategoryResult, error) {
	var out struct {
		Results []CategoryResult `json:"results"`
	}
	err := c.post(ctx, "/v1/categorize", map[string]any{"names": names, "lang": lang}, &out)
	return out.Results, err
}

// ---------- /v1/parse-recipe ----------
type Ingredient struct {
	Raw     string   `json:"raw"`
	Name    *string  `json:"name"`
	Qty     *float64 `json:"qty"`
	Unit    *string  `json:"unit"`
	Section string   `json:"section"`
}
type Recipe struct {
	Title       string       `json:"title"`
	Ingredients []Ingredient `json:"ingredients"`
	Source      string       `json:"source"`
}

func (c *Client) ParseRecipe(ctx context.Context, url, lang string) (*Recipe, error) {
	var out Recipe
	err := c.post(ctx, "/v1/parse-recipe", map[string]any{"url": url, "lang": lang}, &out)
	return &out, err
}

// ---------- /v1/chat (заскоупленный мультиязычный ассистент) ----------
type ChatResult struct {
	Text    string `json:"text"`
	Lang    string `json:"lang"`
	Refused bool   `json:"refused"`
}

func (c *Client) Chat(ctx context.Context, message, lang string) (*ChatResult, error) {
	var out ChatResult
	err := c.post(ctx, "/v1/chat", map[string]any{"message": message, "lang": lang}, &out)
	return &out, err
}

/* Пример:
	aic := ai.New("http://pora-ai:8000")
	preds, _ := aic.Replenishment(ctx, "2026-06-18", purchases)
	cats, _  := aic.Categorize(ctx, []string{"авокадо", "milk"}, userLocale)
	recipe, _ := aic.ParseRecipe(ctx, "https://eda.ru/recipe/carbonara", userLocale)
	answer, _ := aic.Chat(ctx, "что приготовить из курицы?", userLocale)
*/
