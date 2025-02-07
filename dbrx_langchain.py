```python
class GitHubWebhookReceiverView(APIView):
    def post(self, request):
        event = request.headers.get('X-GitHub-Event')
        try:
            if 'payload' in request.data:
                request_data = json.loads(request.data['payload'])
            else:
                request_data = request.data
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON: {e}")
            return HttpResponse(content="error Invalid JSON", status=400)
        try:
            repo_name = request_data['repository']['full_name']
            gh_username = request_data["repository"]["owner"]["login"]
            if event == 'push':
                update_github_PRs_webhook_ctrl_2(request_data, code_only=True)
                update_code_source = update_datasource_repo(
                    UpdateDatasourceReq(
                        Name=repo_name,
                        LatestUpdatedTime=str(timezone.now()),
                        DatasourceType="code",
                        LatestCommitURL=request_data['head_commit']['url']
                    )
                )
                if update_code_source.Error != None:
                    logger.warning("Failed to update issue data source timestamp: " + str(update_code_source.error))
            elif event == 'pull_request':
                if request_data.get('action') == 'closed' and request_data.get('pull_request', {}).get('merged') is True:
                    update_github_PRs_webhook_ctrl_2(request_data)
                    update_pr_source = update_datasource_repo(
                        UpdateDatasourceReq(
                            Name=repo_name,
                            LatestUpdatedTime=str(timezone.now()),
                            DatasourceType="prs",
                            LatestPRNumber=request_data['number']
                        )
                    )
                    if update_pr_source.Error != None:
                        logger.warning("Failed to update issue data source timestamp: " + str(update_pr_source.error))
                else:
                    logger.info("Ignoring non-merged pull request event")
            elif event == 'issues':
                fetch_auth_res = fetch_auth_info_ctrl(
                    FetchAuthInfoRequest(
                        GithubUserName=gh_username, 
                        IntegrationName="github")
                    )
                if fetch_auth_res.Error:
                    logger.warning(f"Failed to update issues due to fetch auth info err for {gh_username}: {fetch_auth_res.Error}")
                else:
                    access_token = fetch_auth_res.Integrations[0].AccessToken
                    update_issue_vector_db(repo_name, access_token=access_token)
                    update_issue_source = update_datasource_repo(
                        UpdateDatasourceReq(
                            Name=repo_name,
                            LatestUpdatedTime=str(timezone.now()),
                            DatasourceType="issues",
                            LatestIssueNumber=request_data['number']
                        )
                    )
                    if update_issue_source.Error != None:
                        logger.warning("Failed to update issue data source timestamp: " + str(update_issue_source.error))
            return HttpResponse(content="OK", status=200)
        except Exception as e:
            logger.info("request_data: " + str(type(request_data)))
            logger.info("request_data: " + str(request_data.keys()))
            logger.error(f"Error processing webhook: {e}", exc_info=1)
            return HttpResponse(content=f"Error - {e}", status=500)
```
