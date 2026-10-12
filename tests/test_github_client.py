import pytest

from unittest.mock import MagicMock, patch

from app.github.client import GitHubApiClient, GitHubApiError, PAGE_SIZE


def _resp(status=200, json_data=None):
    m = MagicMock()
    m.status_code = status
    m.json.return_value = json_data if json_data is not None else []
    m.text = str(json_data or "")
    return m


@pytest.fixture
def client():
    c = GitHubApiClient(token="test-token")
    c.session = MagicMock()
    return c


def test_single_page(client):
    client.session.get.return_value = _resp(json_data=[{"filename": "a.py", "patch": "+x"}])
    files = client.get_pr_files("o/r", 1)
    assert len(files) == 1
    assert files[0]["filename"] == "a.py"
    # Only one call: batch smaller than PAGE_SIZE means last page.
    assert client.session.get.call_count == 1


def test_pagination_collects_all_pages(client):
    page1 = [{"filename": f"f{i}.py"} for i in range(PAGE_SIZE)]
    page2 = [{"filename": "last.py"}]
    client.session.get.side_effect = [_resp(json_data=page1), _resp(json_data=page2)]
    files = client.get_pr_files("o/r", 7)
    assert len(files) == PAGE_SIZE + 1
    assert files[-1]["filename"] == "last.py"
    assert client.session.get.call_count == 2
    # Second call must request page=2
    assert client.session.get.call_args_list[1].kwargs["params"]["page"] == 2


def test_max_files_cap_stops_early(client):
    page1 = [{"filename": f"f{i}.py"} for i in range(PAGE_SIZE + 50)]  # > cap
    client.session.get.return_value = _resp(json_data=page1)
    files = client.get_pr_files("o/r", 7, max_files=10)
    assert len(files) == 10
    assert client.session.get.call_count == 1  # never needed page 2


def test_404_raises_github_api_error(client):
    client.session.get.return_value = _resp(status=404)
    with pytest.raises(GitHubApiError, match="not found"):
        client.get_pr_files("nobody/nowhere", 999)


def test_401_raises_with_token_hint(client):
    client.session.get.return_value = _resp(status=401)
    with pytest.raises(GitHubApiError, match="401"):
        client.get_pr_files("o/r", 1)


def test_403_rate_limit_raises(client):
    client.session.get.return_value = _resp(status=403, json_data={"message": "rate limit"})
    with pytest.raises(GitHubApiError):
        client.get_pr_files("o/r", 1)


def test_token_sent_as_bearer_header(client):
    c = GitHubApiClient(token="abc")
    c.session = MagicMock()
    c.session.get.return_value = _resp(json_data=[])
    c.get_pr_files("o/r", 1)
    sent = c.session.get.call_args.kwargs["headers"]["Authorization"]
    assert sent == "Bearer abc"


def test_no_token_omits_auth_header():
    c = GitHubApiClient(token="")
    c.session = MagicMock()
    c.session.get.return_value = _resp(json_data=[])
    c.get_pr_files("o/r", 1)
    assert "Authorization" not in c.session.get.call_args.kwargs["headers"]


def test_non_list_response_raises(client):
    client.session.get.return_value = _resp(json_data={"message": "unexpected"})
    with pytest.raises(GitHubApiError, match="non-list"):
        client.get_pr_files("o/r", 1)
